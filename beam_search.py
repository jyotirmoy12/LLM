import torch
import torch.nn.functional as F
from typing import Tuple, Set
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from pathlib import Path

from model import Trans
from Embedding import Tokenizer
import evaluate

bleu_metric = evaluate.load("bleu")

class BeamSearchDecoder:
    def __init__(self, model, beam_width: int = 5):
        self.mdl = model
        self.bw = beam_width
    
    def _get_ngrams_fast(self, token_ids: list, n: int) -> Set[Tuple]:
        """Optimized n-gram extraction"""
        if len(token_ids) < n:
            return set()
        return set(tuple(token_ids[i:i+n]) for i in range(len(token_ids) - n + 1))
    
    def _block_ngrams_batch(self, sequences: list, logits: torch.Tensor, 
                            no_repeat_ngram_size: int) -> torch.Tensor:
        """Vectorized n-gram blocking"""
        if no_repeat_ngram_size <= 0:
            return logits
        
        for i, seq in enumerate(sequences):
            if len(seq) < no_repeat_ngram_size:
                continue
            
            # Get existing n-grams
            ngrams = self._get_ngrams_fast(seq, no_repeat_ngram_size)
            prefix_ngrams = self._get_ngrams_fast(seq, no_repeat_ngram_size - 1)
            
            # Block tokens that would create repeated n-grams
            for prefix in prefix_ngrams:
                for tok_id in range(logits.size(-1)):
                    if prefix + (tok_id,) in ngrams:
                        logits[i, tok_id] = float('-inf')
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        eos_token_id: int = 3,
        temperature: float = 1.0,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        early_stopping: bool = True
    ) -> Tuple[torch.Tensor, float]:
        self.mdl.eval()
        device = input_ids.device
        batch_size = 1
        
        # Initialize beams
        beams = [input_ids[0].tolist()]
        beam_scores = torch.zeros(1, device=device)
        
        finished_beams = []
        finished_scores = []
        
        for step in range(max_new_tokens):
            if len(beams) == 0:
                break
            
            if early_stopping and len(finished_beams) >= self.bw:
                break
            
            # Batch all current beams
            beam_tensors = [torch.LongTensor(beam).to(device) for beam in beams]
            max_len = max(len(b) for b in beam_tensors)
            
            # Pad sequences for batching
            padded_beams = []
            for beam in beam_tensors:
                if len(beam) < max_len:
                    padding = torch.zeros(max_len - len(beam), dtype=torch.long, device=device)
                    padded_beams.append(torch.cat([beam, padding]))
                else:
                    padded_beams.append(beam)
            
            batch_input = torch.stack(padded_beams)
            
            # Single batched forward pass (KEY OPTIMIZATION)
            with torch.no_grad():
                logits = self.mdl(batch_input)
                next_logits = logits[:, -1, :] / temperature
            
            # Apply n-gram blocking
            if no_repeat_ngram_size > 0:
                next_logits = self._block_ngrams_batch(beams, next_logits, no_repeat_ngram_size)
            
            # Get log probabilities
            log_probs = F.log_softmax(next_logits, dim=-1)
            
            # Expand beams
            vocab_size = log_probs.size(-1)
            
            # Get top-k tokens for each beam (reduced from bw*2 to bw)
            k = min(self.bw, vocab_size)
            top_log_probs, top_indices = torch.topk(log_probs, k, dim=-1)
            
            # Calculate candidate scores
            candidate_scores = beam_scores.unsqueeze(1) + top_log_probs
            candidate_scores = candidate_scores.view(-1)
            
            # Get top beam_width candidates
            top_cand_scores, top_cand_indices = torch.topk(
                candidate_scores, 
                min(self.bw * 2, len(candidate_scores))  # Reduced multiplier
            )
            
            # Decode indices back to beam and token
            beam_indices = top_cand_indices // k
            token_indices = top_cand_indices % k
            
            new_beams = []
            new_scores = []
            
            for beam_idx, token_idx, score in zip(beam_indices, token_indices, top_cand_scores):
                beam_idx = beam_idx.item()
                token_idx = token_idx.item()
                
                # Get actual token from top_k selection
                actual_token = top_indices[beam_idx, token_idx].item()
                
                new_beam = beams[beam_idx] + [actual_token]
                
                if actual_token == eos_token_id:
                    # Finished beam
                    norm_score = score.item() / (len(new_beam) ** length_penalty)
                    finished_beams.append(new_beam)
                    finished_scores.append(norm_score)
                else:
                    new_beams.append(new_beam)
                    new_scores.append(score.item())
                
                if len(new_beams) >= self.bw:
                    break
            
            beams = new_beams
            beam_scores = torch.tensor(new_scores, device=device)
        
        # Add remaining beams to finished
        for beam, score in zip(beams, beam_scores):
            norm_score = score.item() / (len(beam) ** length_penalty)
            finished_beams.append(beam)
            finished_scores.append(norm_score)
        
        if not finished_beams:
            return input_ids[0], 0.0
        
        best_idx = np.argmax(finished_scores)
        best_beam = torch.LongTensor(finished_beams[best_idx]).to(device)
        
        return best_beam, finished_scores[best_idx]


def eval_beam(model, tok, val_texts, beam_widths=[5], 
              num_samples=5, device='cuda', length_penalty=1.0,
              no_repeat_ngram_size=3):
    results = {}
    eval_configs = [1] + beam_widths
    
    print(f"\n{'='*70}")
    print(f"BEAM SEARCH EVALUATION (OPTIMIZED)")
    print(f"{'='*70}\n")
    
    for bw in eval_configs:
        print(f"\n{'='*70}")
        print(f"Beam Width: {bw}")
        if bw > 1:
            print(f"  length_penalty={length_penalty}, no_repeat_ngram={no_repeat_ngram_size}")
        print(f"{'='*70}")
        
        decoder = BeamSearchDecoder(model, beam_width=bw)
        
        preds = []
        refs = []
        total_tokens = 0
        total_time = 0
        indiv_times = []
        
        eos_idx = tok.w2i['<EOS>']
        
        for idx, text in enumerate(val_texts[:num_samples]):
            token_ids = tok.enc(text)
            
            if len(token_ids) < 6:
                continue
            
            prompt_ids = token_ids[:5]
            reference_continuation_ids = token_ids[5:55]  # Match max_new_tokens=50
            ground_truth = tok.dec(reference_continuation_ids)  # Decode only continuation
            prompt_tensor = torch.LongTensor([prompt_ids]).to(device)
            
            start_time = time.time()
            
            with torch.no_grad():
                if bw == 1:
                    gen = model.generate(
                        prompt_tensor, 
                        max_new_tokens=50, 
                        temperature=1.0,
                        top_k=50,
                        eos_token_id=eos_idx
                    )
                    gen_ids = gen[0].cpu()
                    score = 0.0
                else:
                    gen_ids, score = decoder.generate(
                        prompt_tensor,
                        max_new_tokens=50,
                        eos_token_id=eos_idx,
                        length_penalty=length_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        early_stopping=True
                    )
            
            elapsed = time.time() - start_time
            indiv_times.append(elapsed)
            
            # Decode only the generated continuation (exclude prompt)
            gen_continuation_ids = gen_ids[len(prompt_ids):].tolist()
            gen_txt = tok.dec(gen_continuation_ids)
            preds.append(gen_txt)
            refs.append([ground_truth])  # List of reference strings
            
            num_toks = len(gen_continuation_ids)
            total_tokens += num_toks
            total_time += elapsed
            
            # Diagnostic: print first sample details
            if idx == 0:
                print(f"\n  --- Sample 1 Details ---")
                print(f"  Generated ({len(gen_continuation_ids)} tokens, {len(gen_txt.split())} words): {gen_txt[:100]}...")
                print(f"  Reference ({len(reference_continuation_ids)} tokens, {len(ground_truth.split())} words): {ground_truth[:100]}...")
                print(f"  ------------------------\n")
            
            print(f"  Sample {idx + 1}: {num_toks} tokens, {elapsed:.3f}s, gen_len={len(gen_continuation_ids)}, ref_len={len(reference_continuation_ids)}")
        
        tps = total_tokens / total_time if total_time > 0 else 0
        bleu_res = bleu_metric.compute(predictions=preds, references=refs)
        
        # Diagnostic: Print BLEU components
        print(f"\n  BLEU Components:")
        print(f"    Precisions: {[f'{p:.4f}' for p in bleu_res['precisions']]}")
        print(f"    Brevity Penalty: {bleu_res['brevity_penalty']:.4f}")
        print(f"    Length Ratio: {bleu_res['length_ratio']:.4f}")
        print(f"    Translation Length: {bleu_res['translation_length']}")
        print(f"    Reference Length: {bleu_res['reference_length']}")
        
        results[bw] = {
            'bleu': bleu_res['bleu'],
            'bleu_1': bleu_res['precisions'][0] * 100,
            'bleu_2': bleu_res['precisions'][1] * 100,
            'bleu_3': bleu_res['precisions'][2] * 100,
            'bleu_4': bleu_res['precisions'][3] * 100,
            'tokens_per_second': tps,
            'total_time': total_time,
            'total_tokens': total_tokens,
            'avg_time': np.mean(indiv_times),
            'predictions': preds,
            'references': refs
        }
        
        print(f"\n  Results: BLEU={bleu_res['bleu']:.4f}, Speed={tps:.1f} tok/s\n")
    
    return results


def vis_res(results, save_path='beam_search_results'):
    Path(save_path).mkdir(exist_ok=True, parents=True)
    
    bws = sorted(results.keys())
    bleu_scores = [results[bw]['bleu'] for bw in bws]
    tps = [results[bw]['tokens_per_second'] for bw in bws]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # Plot 1: BLEU vs Beam Width
    axes[0].plot(bws, bleu_scores, marker='o', linewidth=2.5, markersize=10, color='#2E86AB')
    axes[0].set_xlabel('Beam Width', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('BLEU Score', fontsize=13, fontweight='bold')
    axes[0].set_title('Output Quality vs Beam Width', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(bws)
    for bw, score in zip(bws, bleu_scores):
        axes[0].text(bw, score + max(bleu_scores)*0.02, f'{score:.4f}', 
                    ha='center', fontsize=10, fontweight='bold')
    
    # Plot 2: Speed vs Beam Width
    axes[1].plot(bws, tps, marker='s', linewidth=2.5, markersize=10, color='#F77F00')
    axes[1].set_xlabel('Beam Width', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Tokens per Second', fontsize=13, fontweight='bold')
    axes[1].set_title('Decoding Speed vs Beam Width', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(bws)
    for bw, tp in zip(bws, tps):
        axes[1].text(bw, tp + max(tps)*0.03, f'{tp:.1f}', 
                    ha='center', fontsize=10, fontweight='bold')
    
    # Plot 3: Speed-Quality Trade-off
    colors = ['#06A77D', '#D62828', '#F77F00', '#9B59B6']
    for i, bw in enumerate(bws):
        axes[2].scatter(tps[i], bleu_scores[i], s=250, alpha=0.7, 
                       color=colors[i % len(colors)], edgecolors='black', linewidth=2)
        axes[2].annotate(f'k={bw}', (tps[i], bleu_scores[i]), 
                        xytext=(10, 10), textcoords='offset points', 
                        fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Tokens per Second', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('BLEU Score', fontsize=13, fontweight='bold')
    axes[2].set_title('Speed-Quality Trade-off', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_file = Path(save_path) / 'beam_search_comparison.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Visualization saved to {save_file}")
    return fig


def print_res_tbl(results):
    print(f"\n{'='*90}")
    print(f"{'':>28}BEAM SEARCH EVALUATION RESULTS")
    print(f"{'='*90}\n")
    
    print(f"{'Beam':<8} {'BLEU':<10} {'BLEU-1':<10} {'BLEU-4':<10} {'Speed (tok/s)':<15} {'Time (s)':<10}")
    print(f"{'-'*90}")
    
    bws = sorted(results.keys())
    
    for bw in bws:
        res = results[bw]
        beam_label = "Greedy" if bw == 1 else str(bw)
        print(f"{beam_label:<8} {res['bleu']:<10.4f} {res['bleu_1']:<10.2f} {res['bleu_4']:<10.2f} "
              f"{res['tokens_per_second']:<15.1f} {res['total_time']:<10.2f}")
    
    print(f"{'='*90}\n")
    
    # Analysis
    if len(bws) > 1:
        print(f"{'='*90}")
        print(f"{'':>38}ANALYSIS")
        print(f"{'='*90}\n")
        
        best_bw = max(bws, key=lambda x: results[x]['bleu'])
        improvement = ((results[best_bw]['bleu'] - results[1]['bleu']) / results[1]['bleu'] * 100)
        speedup = results[1]['tokens_per_second'] / results[best_bw]['tokens_per_second']
        
        print(f"  Best Beam Width: {best_bw}")
        print(f"    • BLEU Score: {results[best_bw]['bleu']:.4f}")
        print(f"    • BLEU improvement over greedy: {improvement:+.1f}%")
        print(f"    • Speed reduction: {speedup:.1f}x slower than greedy")
        
        print(f"\n  Greedy Search (k=1):")
        print(f"    • BLEU Score: {results[1]['bleu']:.4f}")
        print(f"    • Speed: {results[1]['tokens_per_second']:.1f} tokens/s")
        
        print(f"\n  Key Findings:")
        print(f"    • Beam search improves quality by exploring multiple paths")
        print(f"    • Trade-off: Better quality at the cost of slower inference")
        print(f"    • Optimal beam width balances quality and speed")
        
        print(f"\n{'='*90}\n")


def find_files():
    current_dir = Path.cwd()
    
    model_patterns = ['final_model_complete.pt', 'best_model.pt', '*model*.pt']
    model_path = None
    for pattern in model_patterns:
        matches = list(current_dir.glob(pattern))
        if matches:
            model_path = matches[0]
            break
    
    data_patterns = ['Dataset/val_tokenized.pkl', 'val_tokenized.pkl', '*val*.pkl']
    data_path = None
    for pattern in data_patterns:
        matches = list(current_dir.glob(pattern))
        if matches:
            data_path = matches[0]
            break
    
    tokenizer_patterns = ['Dataset/tokenizer.pkl', 'tokenizer.pkl', '*tokenizer*.pkl']
    tokenizer_path = None
    for pattern in tokenizer_patterns:
        matches = list(current_dir.glob(pattern))
        if matches:
            tokenizer_path = matches[0]
            break
    
    return model_path, data_path, tokenizer_path


def main():
    parser = argparse.ArgumentParser(description='Optimized Beam Search Evaluation')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--val_data', type=str, default=None, help='Path to validation data')
    parser.add_argument('--tokenizer', type=str, default=None, help='Path to tokenizer')
    parser.add_argument('--beam_widths', nargs='+', type=int, default=[5, 10], 
                        help='Beam widths to test')
    parser.add_argument('--num_samples', type=int, default=5, 
                        help='Number of prompts from validation data')
    parser.add_argument('--length_penalty', type=float, default=1.0, 
                        help='Length penalty for beam search')
    parser.add_argument('--no_repeat_ngram', type=int, default=3, 
                        help='Block repeat n-grams')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='beam_search_results', 
                        help='Directory to save results')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"{'':>18}OPTIMIZED BEAM SEARCH EXPERIMENT")
    print(f"{'='*70}\n")
    
    # Auto-find files if not provided
    if args.checkpoint is None or args.val_data is None or args.tokenizer is None:
        print("Searching for required files...")
        model_path, data_path, tokenizer_path = find_files()
        
        args.checkpoint = args.checkpoint or model_path
        args.val_data = args.val_data or data_path
        args.tokenizer = args.tokenizer or tokenizer_path
    
    if args.checkpoint is None or args.val_data is None or args.tokenizer is None:
        print("\n✗ Error: Could not find required files!")
        print("  Please specify: --checkpoint, --val_data, --tokenizer")
        return
    
    print(f"Configuration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Validation data: {args.val_data}")
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"  Beam widths: {args.beam_widths}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Length penalty: {args.length_penalty}")
    print(f"  No-repeat n-gram: {args.no_repeat_ngram}")
    print(f"  Device: {args.device}")
    
    # Load tokenizer
    print(f"\n[1/3] Loading tokenizer...")
    tok = Tokenizer.ld(args.tokenizer)
    vocab_size = len(tok.w2i)
    pad_idx = tok.w2i['<PAD>']
    print(f"  ✓ Vocabulary size: {vocab_size:,}")
    
    # Load validation data
    print(f"\n[2/3] Loading validation data...")
    with open(args.val_data, 'rb') as f:
        val_token_ids = pickle.load(f)
    
    val_texts = [tok.dec(ids) for ids in val_token_ids[:args.num_samples * 3]
                 if len(tok.dec(ids).split()) > 5][:args.num_samples * 2]
    print(f"  ✓ Validation samples: {len(val_texts):,}")
    
    # Load model
    print(f"\n[3/3] Loading model...")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    cfg = ckpt.get('config', {})
    
    model = Trans(
        vocab_size=vocab_size,
        d_model=cfg.get('d_model', 300),
        n_l=cfg.get('num_layers', 3),
        n_heads=cfg.get('num_heads', 10),
        d_ff=cfg.get('d_ff', 2048),
        mx_sq_len=cfg.get('max_seq_len', 64),
        dropout=cfg.get('dropout', 0.3),
        pad_idx=pad_idx
    )
    
    model.load_state_dict(state_dict, strict=True)
    model = model.to(args.device)
    model.eval()
    print(f"  ✓ Model loaded (d_model={cfg.get('d_model', 300)}, layers={cfg.get('num_layers', 3)}, heads={cfg.get('num_heads', 10)})")
    
    # Run evaluation
    results = eval_beam(
        model, tok, val_texts, 
        beam_widths=args.beam_widths,
        num_samples=args.num_samples,
        device=args.device,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram
    )
    
    # Print results
    print_res_tbl(results)
    
    # Visualize results
    vis_res(results, save_path=args.save_dir)
    
    # Save results
    Path(args.save_dir).mkdir(exist_ok=True, parents=True)
    with open(Path(args.save_dir) / 'beam_search_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Results saved to {args.save_dir}/beam_search_results.pkl")
    
    print(f"\n{'='*70}")
    print(f"{'':>25}EXPERIMENT COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()