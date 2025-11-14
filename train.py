import torch
import numpy as np
from torch.utils.data import DataLoader
import pickle
import os
os.environ['WANDB_API_KEY'] = '529ead6048351d29e4104e98fd18ebe1431687d6'
os.environ["HTTP_PROXY"] = "http://EEZ258127:your_password@10.10.78.61:3128"
os.environ["HTTPS_PROXY"] = "http://EEZ258127:your_password@10.10.78.61:3128"
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_DIR'] = './wandb_logs'

from model import Trans
from utils import Trainer, TSDset, collate_fn
from logger import ExperimentLogger
from utils import load_data


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_token_ids, val_token_ids, tokenizer, embedding_matrix, vocab_size = load_data()
    
    config = {
    'vocab_size': vocab_size,
    'd_model': 300,  
    'num_layers': 3,  
    'num_heads': 10,   
    'd_ff': 2048,
    'max_seq_len': 64,
    'dropout': 0.3,
    'batch_size': 64,
    'num_epochs': 10,
    'learning_rate': 3e-4,  
    'pad_idx': tokenizer.w2i['<PAD>'],
    'eos_idx': tokenizer.w2i['<EOS>'],
    'sos_idx': tokenizer.w2i['<SOS>'],  
    'unk_idx': tokenizer.w2i['<UNK>'],  
    'freeze_embeddings': False,
    'use_wandb': True
}
    
    experiment_name = f"exp_L{config['num_layers']}_d{config['d_model']}_h{config['num_heads']}_lr{config['learning_rate']:.0e}"
    logger = ExperimentLogger(config, experiment_name)
    
  
    train_subset = train_token_ids[:20000]
    val_subset = val_token_ids[:2000]
    
    train_dataset = TSDset(train_subset, config['max_seq_len'])
    val_dataset = TSDset(val_subset, config['max_seq_len'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, config['pad_idx']),
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, config['pad_idx']),
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    model = Trans(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_l=config['num_layers'],
        n_heads=config['num_heads'],
        d_ff=config['d_ff'],
        mx_sq_len=config['max_seq_len'],
        dropout=config['dropout'],
        pad_idx=config['pad_idx']
    ).to(device)
    
    model.ld_pre_emb(embedding_matrix, freeze=config['freeze_embeddings'])
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    trainer = Trainer(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        device, 
        config['pad_idx'],
        use_wandb=config['use_wandb'],
        config=config,
        logger=logger
    )
    trainer.train(config['num_epochs'])
    
    torch.save(model.state_dict(), 'final_model.pt')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab_size': vocab_size
    }, 'final_model_complete.pt')
    
    with open('model_config.pkl', 'wb') as f:
        pickle.dump(config, f)
    
if __name__ == "__main__":
    main()