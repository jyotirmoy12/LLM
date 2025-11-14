import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import copy
import argparse
from utils import TSDset, collate_fn


def train_with_accum(model, dataloader, optimizer, accum_steps, 
                     epoch, device, pad_idx):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    batch_losses = []
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        ii = batch['input_ids'].to(device)
        tgi = batch['target_ids'].to(device)
        
        lg = model(ii)
        
        ls = F.cross_entropy(
            lg.view(-1, lg.size(-1)),
            tgi.view(-1),
            ignore_index=pad_idx,
            reduction='mean'
        )
        
        ls = ls / accum_steps
        ls.backward()
        
        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        batch_loss = ls.item() * accum_steps
        total_loss += batch_loss * tgi.numel()
        total_tokens += tgi.numel()
        batch_losses.append(batch_loss)
        
        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / total_tokens
            print(f"    Batch {batch_idx + 1}/{len(dataloader)} | Loss: {avg_loss:.4f}")
    
    if len(dataloader) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / total_tokens
    return avg_loss, batch_losses


def run_grad_accum_exp(
    base_model,
    train_data,
    batch_size=16,
    accum_steps_list=[1, 2, 4, 8],
    num_epochs=3,
    device='cuda',
    pad_idx=0,
    save_dir='grad_accum_results'
):
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    results = {}
    
    for accum_steps in accum_steps_list:
        eff_bs = batch_size * accum_steps
        
        print(f"\n{'='*70}")
        print(f"Configuration: Accumulation Steps = {accum_steps}")
        print(f"  Mini-batch Size: {batch_size}")
        print(f"  Effective Batch Size: {eff_bs}")
        print(f"{'='*70}")
      
        # Create independent model copy for this experiment
        model_copy = copy.deepcopy(base_model)
        model_copy = model_copy.to(device)
        
        optimizer = torch.optim.AdamW(model_copy.parameters(), lr=3e-4, weight_decay=0.01)
        
        dset = TSDset(train_data, mx_sq_len=64)
        dldr = DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, pad_idx=pad_idx),
            num_workers=0,
            pin_memory=True if device == 'cuda' else False
        )
        
        epoch_losses = []
        epoch_times = []
        all_batch_losses = []
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n  Epoch {epoch}/{num_epochs}")
            start_time = time.time()
            
            avg_loss, batch_losses = train_with_accum(
                model_copy, dldr, optimizer, accum_steps,
                epoch, device, pad_idx
            )
            
            epoch_time = time.time() - start_time
            
            epoch_losses.append(avg_loss)
            epoch_times.append(epoch_time)
            all_batch_losses.extend(batch_losses)
            
            print(f"  → Epoch {epoch} Complete | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
        
        results[accum_steps] = {
            'effective_batch_size': eff_bs,
            'epoch_losses': epoch_losses,
            'epoch_times': epoch_times,
            'all_batch_losses': all_batch_losses,
            'avg_epoch_time': sum(epoch_times) / len(epoch_times),
            'final_loss': epoch_losses[-1]
        }
        
        print(f"\n  Summary for Accum Steps = {accum_steps}:")
        print(f"    Final Loss: {epoch_losses[-1]:.4f}")
        print(f"    Avg Epoch Time: {results[accum_steps]['avg_epoch_time']:.2f}s")
    
    return results


def plot_res(results, save_dir):
    # Plot 1: Loss Curves
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#E63946', '#F77F00', '#06A77D', '#4361EE']
    markers = ['o', 's', '^', 'D']
    
    for idx, (accum_steps, res) in enumerate(sorted(results.items())):
        epochs = list(range(1, len(res['epoch_losses']) + 1))
        eff_bs = res['effective_batch_size']
        
        if accum_steps == 1:
            label = f"Baseline (No Accumulation, BS={eff_bs})"
        else:
            label = f"Accum Steps={accum_steps} (Eff. BS={eff_bs})"
        
        ax.plot(epochs, res['epoch_losses'], 
               marker=markers[idx % len(markers)],
               label=label, linewidth=2.5, markersize=10,
               color=colors[idx % len(colors)])
        
        for x, y in zip(epochs, res['epoch_losses']):
            ax.text(x, y + 0.02, f'{y:.3f}', ha='center', va='bottom', 
                   fontsize=8, color=colors[idx % len(colors)])
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=14, fontweight='bold')
    ax.set_title('Training Loss: Gradient Accumulation Comparison', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Loss curves saved to {save_dir}/loss_curves.png")
    
    # Plot 2: Runtime Comparison
    fig, ax = plt.subplots(figsize=(11, 7))
    
    accum_steps_list = sorted(results.keys())
    avg_times = [results[steps]['avg_epoch_time'] for steps in accum_steps_list]
    eff_bs = [results[steps]['effective_batch_size'] for steps in accum_steps_list]
    
    x_pos = range(len(accum_steps_list))
    bars = ax.bar(x_pos, avg_times, 
                  color=colors[:len(accum_steps_list)], alpha=0.7, 
                  edgecolor='black', linewidth=2, width=0.6)
    
    for bar, time_val in zip(bars, avg_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{time_val:.2f}s',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Configuration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Epoch Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Runtime per Epoch: Gradient Accumulation Comparison', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    
    labels = []
    for steps, eff in zip(accum_steps_list, eff_bs):
        if steps == 1:
            labels.append(f'Baseline\n(BS={eff})')
        else:
            labels.append(f'Accum={steps}\n(Eff. BS={eff})')
    
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'runtime_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Runtime comparison saved to {save_dir}/runtime_comparison.png")
    
    # Plot 3: Combined Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    final_losses = [results[steps]['final_loss'] for steps in accum_steps_list]
    bars1 = ax1.bar(x_pos, final_losses, color=colors[:len(accum_steps_list)], 
                    alpha=0.7, edgecolor='black', linewidth=2, width=0.6)
    
    for bar, loss_val in zip(bars1, final_losses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss_val:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('Configuration', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Final Loss', fontsize=13, fontweight='bold')
    ax1.set_title('Final Loss Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    ax2.scatter(avg_times, final_losses, s=300, c=colors[:len(accum_steps_list)],
               alpha=0.6, edgecolors='black', linewidth=2)
    
    for i, (steps, time_val, loss_val) in enumerate(zip(accum_steps_list, avg_times, final_losses)):
        label = f'Accum={steps}' if steps > 1 else 'Baseline'
        ax2.annotate(label, (time_val, loss_val), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3))
    
    ax2.set_xlabel('Avg Epoch Time (seconds)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Final Loss', fontsize=13, fontweight='bold')
    ax2.set_title('Time-Loss Trade-off', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'combined_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Combined analysis saved to {save_dir}/combined_analysis.png")


def print_res_tbl(results):
    print(f"\n{'='*90}")
    print(f"GRADIENT ACCUMULATION EXPERIMENT RESULTS")
    print(f"{'='*90}\n")
    
    print(f"{'Config':<20} | {'Mini-BS':<10} | {'Eff. BS':<10} | {'Final Loss':<12} | {'Avg Time/Epoch':<15}")
    print(f"{'-'*90}")
    
    baseline_time = results[1]['avg_epoch_time']
    
    for accum_steps in sorted(results.keys()):
        res = results[accum_steps]
        cfg = "Baseline" if accum_steps == 1 else f"Accumulation={accum_steps}"
        mini_bs = 16
        eff_bs = res['effective_batch_size']
        final_loss = res['final_loss']
        avg_time = res['avg_epoch_time']
        
        print(f"{cfg:<20} | {mini_bs:<10} | {eff_bs:<10} | {final_loss:<12.4f} | {avg_time:<15.2f}s")
    
    print(f"\n{'='*90}")
    print(f"DETAILED ANALYSIS")
    print(f"{'='*90}")
    
    print("\n  Loss Convergence:")
    for accum_steps in sorted(results.keys()):
        res = results[accum_steps]
        cfg = "Baseline" if accum_steps == 1 else f"Accum={accum_steps}"
        improvement = ((results[1]['final_loss'] - res['final_loss']) / results[1]['final_loss'] * 100)
        print(f"    • {cfg:15s}: Final Loss = {res['final_loss']:.4f} "
              f"({improvement:+.1f}% vs baseline)")
    
    print("\n  Runtime Analysis:")
    for accum_steps in sorted(results.keys()):
        res = results[accum_steps]
        cfg = "Baseline" if accum_steps == 1 else f"Accum={accum_steps}"
        overhead = ((res['avg_epoch_time'] - baseline_time) / baseline_time * 100)
        print(f"    • {cfg:15s}: {res['avg_epoch_time']:.2f}s per epoch "
              f"({overhead:+.1f}% vs baseline)")
    
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
    parser = argparse.ArgumentParser(description='Gradient Accumulation Experiment')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--train_data', type=str, default=None, help='Path to training data')
    parser.add_argument('--tokenizer', type=str, default=None, help='Path to tokenizer')
    parser.add_argument('--batch_size', type=int, default=16, help='Mini-batch size (fixed)')
    parser.add_argument('--accum_steps', type=int, nargs='+', default=[1, 2, 4, 8], 
                        help='Gradient accumulation steps to test')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs per experiment')
    parser.add_argument('--data_subset', type=int, default=5000, 
                        help='Number of samples to use (for speed). Set to -1 to use all data')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='grad_accum_results', help='Directory to save results')
    args = parser.parse_args()
    
    # Auto-find files if not provided
    if args.checkpoint is None or args.train_data is None or args.tokenizer is None:
        print("Searching for required files...")
        model_path, data_path, tokenizer_path = find_files()
        
        args.checkpoint = args.checkpoint or model_path
        args.train_data = args.train_data or data_path
        args.tokenizer = args.tokenizer or tokenizer_path
    
    if args.checkpoint is None or args.train_data is None or args.tokenizer is None:
        print("\n❌ Error: Could not find required files!")
        print("Please provide:")
        print("  --checkpoint: path to model checkpoint")
        print("  --train_data: path to training data")
        print("  --tokenizer: path to tokenizer")
        return
    
    print(f"\n{'='*70}")
    print(f"GRADIENT ACCUMULATION EXPERIMENT CONFIGURATION")
    print(f"{'='*70}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Train data: {args.train_data}")
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"  Mini-batch size (fixed): {args.batch_size}")
    print(f"  Accumulation steps to test: {args.accum_steps}")
    print(f"  Epochs per experiment: {args.num_epochs}")
    print(f"  Device: {args.device}")
    print(f"  Results directory: {args.save_dir}")
    
    # Import your model classes
    from Embedding import Tokenizer
    from model import Trans
    
    # Load tokenizer
    tok = Tokenizer.ld(args.tokenizer)
    vocab_size = len(tok.w2i)
    pad_idx = tok.w2i['<PAD>']
    
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  PAD index: {pad_idx}")
    
    # Load training data
    with open(args.train_data, 'rb') as f:
        train_token_ids = pickle.load(f)
    
    original_size = len(train_token_ids)
    
    # USE SUBSET OF DATA FOR SPEED
    if args.data_subset > 0 and args.data_subset < original_size:
        train_token_ids = train_token_ids[:args.data_subset]
        print(f"  Training samples: {len(train_token_ids)} (subset of {original_size} for speed)")
    else:
        print(f"  Training samples: {len(train_token_ids)} (full dataset)")
    
    print(f"{'='*70}\n")
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    
    if isinstance(ckpt, dict):
        state_dict = ckpt.get('model_state_dict', ckpt)
        cfg = ckpt.get('config', {})
    else:
        state_dict = ckpt
        cfg = {}
    
    # Create base model
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
    print("✓ Model loaded successfully\n")
    
    # Run experiments
    results = run_grad_accum_exp(
        base_model=base_model,
        train_data=train_token_ids,
        batch_size=args.batch_size,
        accum_steps_list=args.accum_steps,
        num_epochs=args.num_epochs,
        device=args.device,
        pad_idx=pad_idx,
        save_dir=args.save_dir
    )
    
    # Print results table
    print_res_tbl(results)
    
    # Generate plots
    plot_res(results, args.save_dir)
    
    # Save results
    with open(Path(args.save_dir) / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Results saved to {args.save_dir}/results.pkl")
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE!")
    print(f"{'='*70}\n")
    
if __name__ == '__main__':
    main()