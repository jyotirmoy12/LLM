import json
import csv
import os
from pathlib import Path
from datetime import datetime
import torch


class ExperimentLogger:
    
    def __init__(self, config, experiment_name=None):
        self.cfg = config
        self.nm = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        Path('experiment_logs').mkdir(exist_ok=True)
        
        self.jp = f'experiment_logs/{self.nm}_config.json'
        self.cp = f'experiment_logs/{self.nm}_results.csv'
        self.scp = 'experiment_logs/all_experiments_summary.csv'
        
        self.sv_cfg()
        self.init_csv()
        self.ep_res = []
        self.st = datetime.now()
    
    def sv_cfg(self):
        cd = {
            'experiment_name': self.nm,
            'timestamp': datetime.now().isoformat(),
            'config': self.cfg,
            'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU'
        }
        
        with open(self.jp, 'w') as f:
            json.dump(cd, f, indent=4)

    def init_csv(self):
        hd = [
            'epoch', 'train_loss', 'val_loss', 
            'train_ppl', 'val_ppl',
            'timestamp'
        ]
        
        with open(self.cp, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(hd)
    
    def log_ep(self, epoch, train_loss, val_loss, train_ppl, val_ppl):
        rw = [
            epoch + 1,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            f"{train_ppl:.4f}",
            f"{val_ppl:.4f}",
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]
        
        with open(self.cp, 'a', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(rw)
        
        self.ep_res.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_ppl': train_ppl,
            'val_ppl': val_ppl
        })
    
    def sv_final(self, best_val_loss=None, total_epochs=0):
        with open(self.jp, 'r') as f:
            cd = json.load(f)
        
        tt = (datetime.now() - self.st).total_seconds()
        hrs = int(tt // 3600)
        mins = int((tt % 3600) // 60)
        
        cd['results'] = {
            'total_epochs': total_epochs,
            'best_val_loss': best_val_loss,
            'final_train_loss': self.ep_res[-1]['train_loss'] if self.ep_res else None,
            'final_val_loss': self.ep_res[-1]['val_loss'] if self.ep_res else None,
            'final_train_ppl': self.ep_res[-1]['train_ppl'] if self.ep_res else None,
            'final_val_ppl': self.ep_res[-1]['val_ppl'] if self.ep_res else None,
            'total_training_time': f"{hrs}h {mins}m",
            'completion_timestamp': datetime.now().isoformat()
        }
        
        with open(self.jp, 'w') as f:
            json.dump(cd, f, indent=4)
        
        self.upd_sum(cd)
    
    def upd_sum(self, cd):
        se = os.path.exists(self.scp)
        
        rw = {
            'experiment_name': self.nm,
            'timestamp': cd['timestamp'],
            'd_model': self.cfg.get('d_model'),
            'num_layers': self.cfg.get('num_layers'),
            'num_heads': self.cfg.get('num_heads'),
            'd_ff': self.cfg.get('d_ff'),
            'dropout': self.cfg.get('dropout'),
            'batch_size': self.cfg.get('batch_size'),
            'learning_rate': self.cfg.get('learning_rate'),
            'max_seq_len': self.cfg.get('max_seq_len'),
            'total_epochs': cd['results']['total_epochs'],
            'best_val_loss': cd['results']['best_val_loss'],
            'final_val_loss': cd['results']['final_val_loss'],
            'final_val_ppl': cd['results']['final_val_ppl'],
            'training_time': cd['results']['total_training_time']
        }
        
        with open(self.scp, 'a', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=rw.keys())
            if not se:
                wr.writeheader()
            wr.writerow(rw)