import math
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from model import Trans
from Embedding import Tokenizer
import evaluate

bleu_metric = evaluate.load("bleu")


def load_prep_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def cmp_ppl_per_tok(model, token_ids, device, pad_idx=0):
    model.eval()
    
    if len(token_ids) < 2:
        return float('inf')
    
    inp = torch.LongTensor([token_ids[:-1]]).to(device)
    tgt = torch.LongTensor([token_ids[1:]]).to(device)
    
    with torch.no_grad():
        lg = model(inp)
        ls = F.cross_entropy(lg.view(-1, lg.size(-1)), tgt.view(-1),
                            ignore_index=pad_idx, reduction='mean')
    
    return math.exp(min(ls.item(), 100))


def extr_attn_wts(model, token_ids, device):
    model.eval()
    inp_tensor = torch.LongTensor([token_ids]).to(device)
    
    try:
        with torch.no_grad():
            lg, attn_wts = model(inp_tensor, ret_attn=True)
        attn_wts = [attn[0].cpu() for attn in attn_wts]
        return attn_wts
    except:
        return []


def eval_on_val(model, tok, val_texts, num_samples=50, device='cuda', pad_idx=0):
    ppls = []
    preds = []
    refs = []
    attn_samples = []
    
    samp_texts = val_texts[:num_samples] if isinstance(val_texts, list) else list(val_texts)[:num_samples]
    
    for idx, text in enumerate(tqdm(samp_texts, desc="Evaluation")):
        token_ids = tok.enc(text)
        
        if len(token_ids) < 6:
            continue
        
        prompt_ids = token_ids[:5]
        ground_truth = tok.dec(token_ids)
        
        try:
            prompt_tensor = torch.LongTensor([prompt_ids]).to(device)
            
            with torch.no_grad():
                gen = model.generate(prompt_tensor, max_new_tokens=50, temperature=0.8,
                                    top_k=50, eos_token_id=tok.w2i['<EOS>'])
            
            gen_ids = gen[0].cpu().tolist()
            gen_text = tok.dec(gen_ids)
            
            ppl = cmp_ppl_per_tok(model, gen_ids, device, pad_idx)
            if not math.isinf(ppl) and not math.isnan(ppl):
                ppls.append(ppl)
            
            preds.append(gen_text)
            refs.append([ground_truth])
            
            if idx < 3:
                attn_samples.append({
                    'prompt': tok.dec(prompt_ids),
                    'generated': gen_text,
                    'token_ids': gen_ids[:15],
                    'ground_truth': ground_truth
                })
        except Exception as e:
            continue
    
    avg_ppl = np.mean(ppls) if ppls else float('inf')
    bleu_sc = 0.0
    
    if len(preds) > 0:
        bleu_res = bleu_metric.compute(predictions=preds, references=refs)
        bleu_sc = bleu_res['bleu']
        
        print(f"\nBLEU-1: {bleu_res['precisions'][0]:.4f}")
        print(f"BLEU-2: {bleu_res['precisions'][1]:.4f}")
        print(f"BLEU-3: {bleu_res['precisions'][2]:.4f}")
        print(f"BLEU-4: {bleu_res['precisions'][3]:.4f}")
    
    print(f"\nSamples: {len(samp_texts)}")
    print(f"Perplexity: {avg_ppl:.2f}")
    print(f"BLEU: {bleu_sc:.2f}")
    
    return {
        'avg_perplexity': avg_ppl,
        'bleu_score': bleu_sc,
        'predictions': preds,
        'references': refs,
        'attention_samples': attn_samples,
        'perplexities': ppls
    }


def gen_samp_outs(model, tok, val_texts, device='cuda'):
    """Generate 5 sample outputs with diverse prompts using stochastic sampling"""
    
    # Select 5 diverse texts
    diverse_texts = []
    seen_starts = set()
    
    for text in val_texts:
        token_ids = tok.enc(text)
        if len(token_ids) < 10:
            continue
        
        # Get first 3 tokens to check diversity
        start_key = tuple(token_ids[:3])
        
        if start_key not in seen_starts:
            diverse_texts.append(text)
            seen_starts.add(start_key)
        
        if len(diverse_texts) >= 5:
            break
    
    # If we don't have 5 diverse texts, just use first 5
    if len(diverse_texts) < 5:
        diverse_texts = [text for text in val_texts if len(tok.enc(text)) >= 10][:5]
    
    print(f"\n{'='*100}")
    print(f"{'':>40}SAMPLE GENERATIONS (STOCHASTIC)")
    print(f"{'='*100}")
    print(f"Note: Using first 5 tokens as prompt with stochastic sampling (temperature & top-k)")
    print(f"{'='*100}\n")
    
    for i, text in enumerate(diverse_texts[:5]):
        token_ids = tok.enc(text)
        
        if len(token_ids) < 10:
            continue
        
        # Use EXACTLY first 5 tokens as prompt
        prompt_ids = token_ids[:5]
        prompt_text = tok.dec(prompt_ids)
        
        print(f"\n{'-'*100}")
        print(f"Sample {i+1}/5")
        print(f"{'-'*100}")
        print(f"Prompt (5 tokens): {prompt_text}")
        print(f"Ground Truth: {text[:150]}...")
        print(f"\nStochastic Generations (Non-Deterministic):")
        
        # Generate with different stochastic strategies
        strategies = [
            {'temp': 0.7, 'top_k': 50, 'desc': 'Conservative (temp=0.7, top_k=50)'},
            {'temp': 1.0, 'top_k': 40, 'desc': 'Balanced (temp=1.0, top_k=40)'},
            {'temp': 1.2, 'top_k': 50, 'desc': 'Creative (temp=1.2, top_k=50)'},
        ]
        
        for strategy in strategies:
            try:
                prompt_tensor = torch.LongTensor([prompt_ids]).to(device)
                with torch.no_grad():
                    gen = model.generate(
                        prompt_tensor, 
                        max_new_tokens=50, 
                        temperature=strategy['temp'],
                        top_k=strategy['top_k'],
                        eos_token_id=tok.w2i['<EOS>']
                    )
                gen_text = tok.dec(gen[0].cpu().tolist())
                print(f"  {strategy['desc']}:")
                print(f"    {gen_text[:130]}...")
            except Exception as e:
                print(f"  {strategy['desc']}: [Generation failed: {e}]")
        print()


def vis_all_hds_lyr(attn_wts, tokens, lyr_idx, save_dir):
    num_hds = attn_wts.size(0)
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f'Layer {lyr_idx} - All Attention Heads', fontsize=18, fontweight='bold')
    
    for hd_idx in range(min(num_hds, 8)):
        row = hd_idx // 4
        col = hd_idx % 4
        ax = axes[row, col]
        attn = attn_wts[hd_idx].numpy()
        
        im = ax.imshow(attn, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_title(f'Head {hd_idx}', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    save_path = Path(save_dir) / f'layer_{lyr_idx}_all_heads.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_attn_pats(model, tok, attn_samples, save_dir='visualizations', device='cuda'):
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*100}")
    print(f"Analyzing attention patterns...")
    print(f"{'='*100}\n")
    
    for idx, samp in enumerate(attn_samples):
        try:
            attn_wts = extr_attn_wts(model, samp['token_ids'], device)
            
            if not attn_wts:
                continue
            
            tokens = [tok.i2w.get(tid, '<UNK>') for tid in samp['token_ids']]
            tokens = [t.replace('</w>', '') for t in tokens]
            samp_dir = Path(save_dir) / f'sample_{idx + 1}'
            samp_dir.mkdir(exist_ok=True)
            
            for lyr_idx, lyr_attn in enumerate(attn_wts):
                vis_all_hds_lyr(lyr_attn, tokens, lyr_idx, samp_dir)
            
            print(f"  ✓ Attention visualizations saved for sample {idx + 1}")
        except Exception as e:
            print(f"  ✗ Failed to visualize sample {idx + 1}: {e}")


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
    parser = argparse.ArgumentParser(description='Model Testing and Evaluation')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--val_data', type=str, default=None, help='Path to validation data')
    parser.add_argument('--tokenizer', type=str, default=None, help='Path to tokenizer')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='test_results', help='Directory to save results')
    parser.add_argument('--skip_attention', action='store_true', help='Skip attention visualization')
    args = parser.parse_args()
    
    print(f"\n{'='*100}")
    print(f"{'':>42}MODEL TESTING")
    print(f"{'='*100}\n")
    
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
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Device: {args.device}")
    
    # Load tokenizer
    print(f"\n[1/3] Loading tokenizer...")
    tok = Tokenizer.ld(args.tokenizer)
    vocab_size = len(tok.w2i)
    pad_idx = tok.w2i['<PAD>']
    print(f"  ✓ Vocabulary size: {vocab_size:,}")
    
    # Load validation data
    print(f"\n[2/3] Loading validation data...")
    val_token_ids = load_prep_data(args.val_data)
    
    val_texts = []
    for token_ids in val_token_ids[:args.num_samples * 2]:
        text = tok.dec(token_ids)
        if len(text.split()) > 5:
            val_texts.append(text)
    print(f"  ✓ Validation samples: {len(val_texts):,}")
    
    # Load model
    print(f"\n[3/3] Loading model...")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    
    if isinstance(ckpt, dict):
        state_dict = ckpt.get('model_state_dict', ckpt)
        cfg = ckpt.get('config', {})
    else:
        state_dict = ckpt
        cfg = {}
    
    # Use your actual model configuration from train.py
    model = Trans(
        vocab_size=vocab_size,
        d_model=cfg.get('d_model', 300),      # Matches train.py
        n_l=cfg.get('num_layers', 3),         # Matches train.py
        n_heads=cfg.get('num_heads', 10),     # Matches train.py
        d_ff=cfg.get('d_ff', 2048),           # Matches train.py
        mx_sq_len=cfg.get('max_seq_len', 64), # Matches train.py
        dropout=cfg.get('dropout', 0.3),      # Matches train.py
        pad_idx=pad_idx
    )
    
    model.load_state_dict(state_dict, strict=True)
    model = model.to(args.device)
    model.eval()
    print(f"  ✓ Model loaded (d_model={cfg.get('d_model', 300)}, layers={cfg.get('num_layers', 3)}, heads={cfg.get('num_heads', 10)})")
    
    # Create save directory
    Path(args.save_dir).mkdir(exist_ok=True, parents=True)
    
    # Evaluation
    print(f"\n{'='*100}")
    print(f"Running evaluation on {min(args.num_samples, len(val_texts))} samples...")
    print(f"{'='*100}\n")
    
    results = eval_on_val(model, tok, val_texts, 
                         min(args.num_samples, len(val_texts)), args.device, pad_idx)
    
    # Save results
    with open(Path(args.save_dir) / 'evaluation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Results saved to {args.save_dir}/evaluation_results.pkl")
    
    # Generate EXACTLY 5 sample outputs with stochastic sampling
    gen_samp_outs(model, tok, val_texts, args.device)
    
    # Attention visualization
    if not args.skip_attention:
        analyze_attn_pats(model, tok, results['attention_samples'],
                         str(Path(args.save_dir) / 'visualizations'), args.device)
    
    # Final summary
    print(f"\n{'='*100}")
    print(f"{'':>42}FINAL RESULTS")
    print(f"{'='*100}")
    print(f"  Perplexity: {results['avg_perplexity']:.2f}")
    print(f"  BLEU Score: {results['bleu_score']:.4f}")
    print(f"  Samples Generated: {len(results['predictions'])}/{args.num_samples}")
    print(f"  Stochastic Samples Shown: 5")
    print(f"{'='*100}\n")


if __name__ == '__main__':
    main()
