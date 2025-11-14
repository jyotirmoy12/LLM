import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import pickle
import argparse
import copy
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from utils import TSDset, collate_fn


def get_mem_usage(device):
    if isinstance(device, str):
        device_type = device
    else:
        device_type = device.type
    
    if device_type == 'cuda':
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0


def train_one_epoch(model, dataloader, optimizer, device, pad_idx, use_checkpoint=False):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    batch_losses = []
    
    if isinstance(device, str):
        device_obj = torch.device(device)
        device_type = device
    else:
        device_obj = device
        device_type = device.type

    if device_type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    
    for batch_idx, batch in enumerate(dataloader):
        ii = batch['input_ids'].to(device_obj)
        tgi = batch['target_ids'].to(device_obj)
        
        optimizer.zero_grad()
        
        if use_checkpoint and hasattr(model, 'enable_checkpointing'):
            model.enable_checkpointing(True)
        
        lg = model(ii)
        
        if use_checkpoint and hasattr(model, 'enable_checkpointing'):
            model.enable_checkpointing(False)
        
        ls = F.cross_entropy(
            lg.view(-1, lg.size(-1)),
            tgi.view(-1),
            ignore_index=pad_idx,
            reduction='mean'
        )
        
        ls.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        batch_loss = ls.item()
        total_loss += batch_loss * tgi.numel()
        total_tokens += tgi.numel()
        batch_losses.append(batch_loss)
        
        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / total_tokens
            mem_used = get_mem_usage(device_type)
            mem_str = f"Mem: {mem_used:.2f}GB" if device_type == 'cuda' else "Mem: N/A (CPU)"
            print(f"    Batch {batch_idx + 1}/{len(dataloader)} | Loss: {avg_loss:.4f} | {mem_str}")
    
    avg_loss = total_loss / total_tokens
    peak_mem = get_mem_usage(device_type)
    
    return avg_loss, batch_losses, peak_mem


def run_checkpoint_exp(
    base_model,
    train_data,
    batch_size=16,
    num_epochs=3,
    device='cuda',
    pad_idx=0,
    save_dir='checkpoint_results'
):
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    results = {}
    
    if isinstance(device, str):
        device_obj = torch.device(device)
        device_type = device
    else:
        device_obj = device
        device_type = device.type
    
    configs = [
        ('without_checkpoint', False),
        ('with_checkpoint', True)
    ]
    
    for config_name, use_checkpoint in configs:
    
        model_copy = copy.deepcopy(base_model)
        model_copy = model_copy.to(device_obj)
        
        optimizer = torch.optim.AdamW(model_copy.parameters(), lr=3e-4, weight_decay=0.01)
        
        dset = TSDset(train_data, mx_sq_len=64)
        dldr = DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, pad_idx=pad_idx),
            num_workers=0,
            pin_memory=True if device_type == 'cuda' else False
        )
        
        epoch_losses = []
        epoch_times = []
        peak_memories = []
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n  Epoch {epoch}/{num_epochs}")
            start_time = time.time()
            
            avg_loss, batch_losses, peak_mem = train_one_epoch(
                model_copy, dldr, optimizer, device_obj, pad_idx, use_checkpoint
            )
            
            epoch_time = time.time() - start_time
            
            epoch_losses.append(avg_loss)
            epoch_times.append(epoch_time)
            peak_memories.append(peak_mem)
            
            mem_str = f"Peak Mem: {peak_mem:.2f}GB" if device_type == 'cuda' else "Peak Mem: N/A (CPU)"
            print(f" Epoch {epoch} | Loss: {avg_loss:.4f} | "
                  f"Time: {epoch_time:.2f}s | {mem_str}")
        
        results[config_name] = {
            'use_checkpoint': use_checkpoint,
            'epoch_losses': epoch_losses,
            'epoch_times': epoch_times,
            'peak_memories': peak_memories,
            'avg_epoch_time': sum(epoch_times) / len(epoch_times),
            'avg_peak_memory': sum(peak_memories) / len(peak_memories),
            'final_loss': epoch_losses[-1]
        }
        
    
    return results


def plot_res(results, save_dir, is_cuda=True):
    fig, axes = plt.subplots(1, 3 if is_cuda else 2, figsize=(18 if is_cuda else 12, 5))
    
    colors = ['#E63946', '#06A77D']
    config_names = ['without_checkpoint', 'with_checkpoint']
    labels = ['Without Checkpointing', 'With Checkpointing']
    
    # Loss comparison
    ax_idx = 0
    for idx, (config_name, label) in enumerate(zip(config_names, labels)):
        res = results[config_name]
        epochs = list(range(1, len(res['epoch_losses']) + 1))
        
        axes[ax_idx].plot(epochs, res['epoch_losses'], 
                    marker='o', label=label, linewidth=2.5, markersize=10,
                    color=colors[idx])
    
    axes[ax_idx].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[ax_idx].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    axes[ax_idx].set_title('Loss Comparison', fontsize=14, fontweight='bold')
    axes[ax_idx].legend(fontsize=10, loc='best')
    axes[ax_idx].grid(True, alpha=0.3, linestyle='--')
    
    # Runtime comparison
    ax_idx = 1
    configs_bar = ['Without\nCheckpointing', 'With\nCheckpointing']
    times = [results['without_checkpoint']['avg_epoch_time'],
             results['with_checkpoint']['avg_epoch_time']]
    
    bars = axes[ax_idx].bar(configs_bar, times, color=colors, alpha=0.7, 
                      edgecolor='black', linewidth=2, width=0.5)
    
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        axes[ax_idx].text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.2f}s',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    axes[ax_idx].set_ylabel('Average Epoch Time (seconds)', fontsize=12, fontweight='bold')
    axes[ax_idx].set_title('Runtime Comparison', fontsize=14, fontweight='bold')
    axes[ax_idx].grid(True, alpha=0.3, axis='y')
    
    # Memory comparison (only if CUDA)
    if is_cuda:
        ax_idx = 2
        memories = [results['without_checkpoint']['avg_peak_memory'],
                    results['with_checkpoint']['avg_peak_memory']]
        
        bars = axes[ax_idx].bar(configs_bar, memories, color=colors, alpha=0.7,
                          edgecolor='black', linewidth=2, width=0.5)
        
        for bar, mem_val in zip(bars, memories):
            height = bar.get_height()
            axes[ax_idx].text(bar.get_x() + bar.get_width()/2., height,
                        f'{mem_val:.2f}GB',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        if memories[0] > 0:
            mem_saved = memories[0] - memories[1]
            mem_saved_pct = (mem_saved / memories[0]) * 100
            axes[ax_idx].text(0.5, 0.95, f'Memory Saved: {mem_saved_pct:.1f}%',
                        transform=axes[ax_idx].transAxes, ha='center', va='top',
                        fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6),
                        fontweight='bold')
        
        axes[ax_idx].set_ylabel('Peak Memory (GB)', fontsize=12, fontweight='bold')
        axes[ax_idx].set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
        axes[ax_idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.close()


def print_res_tbl(results, is_cuda=True):

    if is_cuda:
        print(f"{'Configuration':<25} | {'Final Loss':<12} | {'Avg Time/Epoch':<15} | {'Avg Peak Mem':<15}")
    else:
        print(f"{'Configuration':<25} | {'Final Loss':<12} | {'Avg Time/Epoch':<15}")

    
    without = results['without_checkpoint']
    with_chk = results['with_checkpoint']
    
    if is_cuda:
        print(f"{'Without Checkpointing':<25} | {without['final_loss']:<12.4f} | "
              f"{without['avg_epoch_time']:<15.2f}s | {without['avg_peak_memory']:<15.2f}GB")
        print(f"{'With Checkpointing':<25} | {with_chk['final_loss']:<12.4f} | "
              f"{with_chk['avg_epoch_time']:<15.2f}s | {with_chk['avg_peak_memory']:<15.2f}GB")
    else:
        print(f"{'Without Checkpointing':<25} | {without['final_loss']:<12.4f} | "
              f"{without['avg_epoch_time']:<15.2f}s")
        print(f"{'With Checkpointing':<25} | {with_chk['final_loss']:<12.4f} | "
              f"{with_chk['avg_epoch_time']:<15.2f}s")
    
    loss_diff = with_chk['final_loss'] - without['final_loss']
    loss_diff_pct = (loss_diff / without['final_loss']) * 100
    
    time_overhead = with_chk['avg_epoch_time'] - without['avg_epoch_time']
    time_overhead_pct = (time_overhead / without['avg_epoch_time']) * 100
    
    if is_cuda and without['avg_peak_memory'] > 0:
        mem_saved = without['avg_peak_memory'] - with_chk['avg_peak_memory']
        mem_saved_pct = (mem_saved / without['avg_peak_memory']) * 100
        



def find_files():
    current_dir = Path.cwd()
    
    model_patterns = ['final_model_complete.pt', 'best_model.pt', '*model*.pt']
    model_path = None
    for pattern in model_patterns:
        matches = list(current_dir.glob(pattern))
        if matches:
            model_path = matches[0]
            break
    
    data_patterns = ['Dataset/train_tokenized.pkl', 'train_tokenized.pkl', 
                     'Dataset/val_tokenized.pkl', '*train*.pkl']
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
    parser = argparse.ArgumentParser(description='Gradient Checkpointing Experiment')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--train_data', type=str, default=None, help='Path to training data')
    parser.add_argument('--tokenizer', type=str, default=None, help='Path to tokenizer')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--data_subset', type=int, default=5000, 
                        help='Number of samples to use (for speed). Set to -1 to use all data')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='checkpoint_results', help='Directory to save results')
    args = parser.parse_args()
    
    
    if args.checkpoint is None or args.train_data is None or args.tokenizer is None:
        model_path, data_path, tokenizer_path = find_files()
        
        args.checkpoint = args.checkpoint or model_path
        args.train_data = args.train_data or data_path
        args.tokenizer = args.tokenizer or tokenizer_path
    
    is_cuda = args.device == 'cuda'
    
    from Embedding import Tokenizer
    from model import Trans
    
    tok = Tokenizer.ld(args.tokenizer)
    vocab_size = len(tok.w2i)
    pad_idx = tok.w2i['<PAD>']

    with open(args.train_data, 'rb') as f:
        train_token_ids = pickle.load(f)
    
    original_size = len(train_token_ids)
    
    if args.data_subset > 0 and args.data_subset < original_size:
        train_token_ids = train_token_ids[:args.data_subset]
    else:
        print(f" Training samples: {len(train_token_ids):,} (full dataset)")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    
    if isinstance(ckpt, dict):
        state_dict = ckpt.get('model_state_dict', ckpt)
        cfg = ckpt.get('config', {})
    else:
        state_dict = ckpt
        cfg = {}
    
    base_model = Trans(
        vocab_size=vocab_size,
        d_model=cfg.get('d_model', 300),      
        n_l=cfg.get('num_layers', 3),         
        n_heads=cfg.get('num_heads', 10),     
        d_ff=cfg.get('d_ff', 2048),           
        mx_sq_len=cfg.get('max_seq_len', 64), 
        dropout=cfg.get('dropout', 0.3),      
        pad_idx=pad_idx
    )
    
    base_model.load_state_dict(state_dict, strict=True)
    

    
    results = run_checkpoint_exp(
        base_model=base_model,
        train_data=train_token_ids,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=args.device,
        pad_idx=pad_idx,
        save_dir=args.save_dir
    )
    
    print_res_tbl(results, is_cuda=is_cuda)
    plot_res(results, args.save_dir, is_cuda=is_cuda)
    
    with open(Path(args.save_dir) / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
if __name__ == '__main__':
    main()