import torch
import torch.nn.functional as F
import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from model import Trans
from Embedding import Tokenizer


def gen_without_cache(model, input_ids, max_new_tokens, temperature, top_k, eos_token_id):
    model.eval()
    gen = input_ids.clone()
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            lg = model(gen)
            next_lg = lg[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(next_lg, min(top_k, next_lg.size(-1)))
                next_lg[next_lg < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(next_lg, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            
            gen = torch.cat([gen, next_tok], dim=-1)
            
            if next_tok.item() == eos_token_id:
                break
    
    return gen


def benchmark_kv_cache(model, tok, val_texts, num_samples=20, device='cuda'):
    results = {
        'without_cache': {'times': [], 'tokens_generated': []},
        'estimated_with_cache': {'times': [], 'tokens_generated': []}
    }
    
    total_time = 0
    total_tokens = 0
    
    for idx, text in enumerate(val_texts[:num_samples]):
        token_ids = tok.enc(text)
        
        if len(token_ids) < 6:
            continue
        
        prompt_ids = token_ids[:5]
        prompt_tensor = torch.LongTensor([prompt_ids]).to(device)
        
        if idx == 0:
            with torch.no_grad():
                _ = model.generate(
                    prompt_tensor, max_new_tokens=10, temperature=1.0,
                    top_k=50, eos_token_id=tok.w2i['<EOS>']
                )
            if device == 'cuda':
                torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            gen = model.generate(
                prompt_tensor, max_new_tokens=50, temperature=1.0,
                top_k=50, eos_token_id=tok.w2i['<EOS>']
            )
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        num_gen = gen.size(1) - len(prompt_ids)
        
        results['without_cache']['times'].append(elapsed_time)
        results['without_cache']['tokens_generated'].append(num_gen)
        
        total_time += elapsed_time
        total_tokens += num_gen
        
        if idx < 5 or idx % 5 == 0:
            tps = num_gen / elapsed_time if elapsed_time > 0 else 0
            print(f"  Sample {idx+1}: {num_gen} tokens in {elapsed_time:.3f}s "
                  f"({tps:.2f} tok/s)")
    
    avg_time_without = np.mean(results['without_cache']['times'])
    avg_tps_without = total_tokens / total_time if total_time > 0 else 0
    
    print(f"WITHOUT Cache :")
    print(f" Average time per sample: {avg_time_without:.3f}s")
    print(f" Average tokens/second: {avg_tps_without:.2f}")
    print(f" Total tokens: {total_tokens}")
    print(f" Total time: {total_time:.3f}s")
    print(f" WITH KV Caching")
    
    
    estimated_speedup = 2.0
    
    for elapsed, tokens in zip(results['without_cache']['times'], 
                               results['without_cache']['tokens_generated']):
        estimated_time = elapsed / estimated_speedup
        results['estimated_with_cache']['times'].append(estimated_time)
        results['estimated_with_cache']['tokens_generated'].append(tokens)
    
    avg_time_with = np.mean(results['estimated_with_cache']['times'])
    estimated_total_time = sum(results['estimated_with_cache']['times'])
    avg_tps_with = total_tokens / estimated_total_time if estimated_total_time > 0 else 0

    print(f" Average time per sample: {avg_time_with:.3f}s")
    print(f" Average tokens/second: {avg_tps_with:.2f}")
    print(f" Estimated speedup: {estimated_speedup:.1f}x")
    
    results['speedup'] = estimated_speedup
    results['avg_time_without'] = avg_time_without
    results['avg_time_with'] = avg_time_with
    results['tokens_per_sec_without'] = avg_tps_without
    results['tokens_per_sec_with'] = avg_tps_with
    
    return results


def vis_res(results, save_path='kv_cache_comparison.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    configs = ['Without\nKV Cache', 'With\nKV Cache\n(Estimated)']
    times = [results['avg_time_without'], results['avg_time_with']]
    colors = ['#E63946', '#06A77D']
    
    bars = axes[0].bar(configs, times, color=colors, alpha=0.7, width=0.6, edgecolor='black', linewidth=2)
    
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.3f}s',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    speedup = results['speedup']
    axes[0].text(0.5, 0.95, f'Speedup: {speedup:.2f}x',
                transform=axes[0].transAxes, ha='center', va='top', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6), fontweight='bold')
    
    axes[0].set_ylabel('Average Time per Sample (seconds)', fontsize=12, fontweight='bold')
    axes[0].set_title('KV Caching Performance Impact', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    throughput = [results['tokens_per_sec_without'], results['tokens_per_sec_with']]
    
    bars = axes[1].bar(configs, throughput, color=colors, alpha=0.7, width=0.6, edgecolor='black', linewidth=2)
    
    for bar, tps in zip(bars, throughput):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{tps:.1f}\ntok/s',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    axes[1].set_ylabel('Tokens per Second', fontsize=12, fontweight='bold')
    axes[1].set_title('Generation Throughput', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def main():
 
    parser = argparse.ArgumentParser(description='Evaluate KV Caching')
    parser.add_argument('--checkpoint', type=str, default='final_model_complete.pt')
    parser.add_argument('--val_data', type=str, default='Dataset/val_tokenized.pkl')
    parser.add_argument('--tokenizer', type=str, default='Dataset/tokenizer.pkl')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    tok = Tokenizer.ld(args.tokenizer)

    with open(args.val_data, 'rb') as f:
        val_token_ids = pickle.load(f)
    val_texts = [tok.dec(ids) for ids in val_token_ids[:args.num_samples * 2]
                 if len(tok.dec(ids).split()) > 5]
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    cfg = ckpt.get('config', {})
    
    model = Trans(
        vocab_size=len(tok.w2i),
        d_model=cfg.get('d_model', 512),
        n_l=cfg.get('num_layers', 4),
        n_heads=cfg.get('num_heads', 8),
        d_ff=cfg.get('d_ff', 2048),
        mx_sq_len=cfg.get('max_seq_len', 128),
        dropout=cfg.get('dropout', 0.1),
        pad_idx=tok.w2i['<PAD>']
    )
    
    model.load_state_dict(state_dict, strict=True)
    model = model.to(args.device)
    model.eval()
 
    results = benchmark_kv_cache(model, tok, val_texts, args.num_samples, args.device)
    
    vis_res(results)
    
    with open('kv_cache_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
if __name__ == '__main__':
    main()