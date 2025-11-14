import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn as nn
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import wandb
import pickle

from Embedding import Tokenizer


class TSDset(Dataset):
    def __init__(self, token_ids_list, mx_sq_len=128):
        self.tl = token_ids_list
        self.msl = mx_sq_len
    
    def __len__(self):
        return len(self.tl)
    
    def __getitem__(self, idx):
        ti = self.tl[idx]
        
        if len(ti) > self.msl + 1:
            ti = ti[:self.msl + 1]

        ii = ti[:-1]
        tg = ti[1:]
        
        return {
            'input_ids': torch.LongTensor(ii),
            'target_ids': torch.LongTensor(tg),
            'length': len(ii)
        }


def collate_fn(batch, pad_idx=0):
    ml = max(sample['length'] for sample in batch)
    
    ii = []
    tg = []
    
    for sample in batch:
        pl = ml - sample['length']
        ii.append(F.pad(sample['input_ids'], (0, pl), value=pad_idx))
        tg.append(F.pad(sample['target_ids'], (0, pl), value=pad_idx))
    
    return {
        'input_ids': torch.stack(ii),
        'target_ids': torch.stack(tg)
    }


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, 
                 pad_idx=0, use_wandb=True, config=None, logger=None):
        self.mdl = model
        self.trl = train_loader
        self.vl = val_loader
        self.opt = optimizer
        self.dev = device
        self.pi = pad_idx
        self.crit = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.log = logger
        
        self.trl_ls = []
        self.val_ls = []
        self.trl_ppl = []
        self.val_ppl = []
        
        self.bvl = float('inf')
        
        if use_wandb:
            wandb.init(
                project="transformer-tinystories",
                config=config,
                name=f"transformer_{config.get('num_layers', 3)}L_{config.get('d_model', 512)}d"
            )
            wandb.watch(model, log='all', log_freq=100)
    
    def tr_ep(self):
        self.mdl.train()
        tl = 0
        nb = 0
        
        pb = tqdm(self.trl, desc="Training")
        for b in pb:
            ii = b['input_ids'].to(self.dev)
            tg = b['target_ids'].to(self.dev)
            
            lg = self.mdl(ii)
            
            ls = self.crit(
                lg.reshape(-1, lg.size(-1)),
                tg.reshape(-1)
            )
            
            self.opt.zero_grad()
            ls.backward()
            torch.nn.utils.clip_grad_norm_(self.mdl.parameters(), max_norm=1.0)
            self.opt.step()
            
            tl += ls.item()
            nb += 1
            pb.set_postfix({'loss': f'{ls.item():.4f}'})
            
            if wandb.run and nb % 10 == 0:
                wandb.log({'train/step_loss': ls.item()})
        
        al = tl / nb
        ppl = math.exp(min(al, 100))
        
        return al, ppl
    
    def val(self):
        self.mdl.eval()
        tl = 0
        nb = 0
        
        with torch.no_grad():
            for b in tqdm(self.vl, desc="Validation"):
                ii = b['input_ids'].to(self.dev)
                tg = b['target_ids'].to(self.dev)
                
                lg = self.mdl(ii)
                
                ls = self.crit(
                    lg.reshape(-1, lg.size(-1)),
                    tg.reshape(-1)
                )
                
                tl += ls.item()
                nb += 1
        
        al = tl / nb
        ppl = math.exp(min(al, 100))
        
        return al, ppl
    
    def train(self, num_epochs):
        for ep in range(num_epochs):
            print(f"\nEpoch {ep + 1}/{num_epochs}")
            
            tr_ls, tr_ppl = self.tr_ep()
            self.trl_ls.append(tr_ls)
            self.trl_ppl.append(tr_ppl)
            
            v_ls, v_ppl = self.val()
            self.val_ls.append(v_ls)
            self.val_ppl.append(v_ppl)
            
            if v_ls < self.bvl:
                self.bvl = v_ls
            
            print(f"Train Loss: {tr_ls:.4f} | Train PPL: {tr_ppl:.2f}")
            print(f"Val Loss:   {v_ls:.4f} | Val PPL:   {v_ppl:.2f}")
            
            if self.log:
                self.log.log_ep(ep, tr_ls, v_ls, tr_ppl, v_ppl)
            
            if wandb.run:
                wandb.log({
                    'epoch': ep + 1,
                    'train/loss': tr_ls,
                    'train/perplexity': tr_ppl,
                    'val/loss': v_ls,
                    'val/perplexity': v_ppl
                })
            
            self.sv_ckpt(ep)
            
            if (ep + 1) % 10 == 0:
                self.plt_met()
        
        if self.log:
            self.log.sv_final(self.bvl, num_epochs)
        
        self.plt_met()
        if wandb.run:
            wandb.finish()
    
    def sv_ckpt(self, epoch):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.mdl.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'train_losses': self.trl_ls,
            'val_losses': self.val_ls,
            'train_perplexities': self.trl_ppl,
            'val_perplexities': self.val_ppl
        }
        
        Path('checkpoints').mkdir(exist_ok=True)
        torch.save(ckpt, f'checkpoints/checkpoint_epoch_{epoch + 1}.pt')
    
    def plt_met(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        eps = range(1, len(self.trl_ls) + 1)
        
        axes[0].plot(eps, self.trl_ls, label='Train Loss', 
                    marker='o', linewidth=2, color='#2E86AB')
        axes[0].plot(eps, self.val_ls, label='Val Loss', 
                    marker='s', linewidth=2, color='#A23B72')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(eps, self.trl_ppl, label='Train PPL', 
                    marker='o', linewidth=2, color='#F18F01')
        axes[1].plot(eps, self.val_ppl, label='Val PPL', 
                    marker='s', linewidth=2, color='#C73E1D')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Perplexity', fontsize=12)
        axes[1].set_title('Training and Validation Perplexity', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
        
        if wandb.run:
            wandb.log({"training_metrics": wandb.Image('training_metrics.png')})
        
        plt.close()

def load_data():
    with open('Dataset/train_tokenized.pkl', 'rb') as f:
        tr_ti = pickle.load(f)
    
    with open('Dataset/val_tokenized.pkl', 'rb') as f:
        v_ti = pickle.load(f)

    tok = Tokenizer.ld('Dataset/tokenizer.pkl')
    vs = len(tok.w2i)
    
    with open('Dataset/embeddings.pkl', 'rb') as f:
        emb_d = pickle.load(f)
    emb_m = emb_d['embeddings']
    
    print(f"Train sequences: {len(tr_ti):,}")
    print(f"Val sequences: {len(v_ti):,}")
    print(f"Vocabulary size: {vs:,}")
    
    return tr_ti, v_ti, tok, emb_m, vs