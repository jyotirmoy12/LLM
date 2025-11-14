import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def emb_l(embs, d_model=512):
    vocab_size, dim = embs.shape
    emb = nn.Embedding(vocab_size, dim, padding_idx=0)
    emb.weight.data = torch.FloatTensor(embs)
    
    if dim != d_model:
        prj = nn.Linear(dim, d_model)
    else:
        prj = nn.Identity()
    
    class PrjEmb(nn.Module):
        def __init__(self, emb, prj):
            super().__init__()
            self.emb = emb
            self.prj = prj
        
        def forward(self, x):
            return self.prj(self.emb(x))
    
    return PrjEmb(emb, prj)


class LN(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        m = x.mean(dim=-1, keepdim=True)
        s = x.std(dim=-1, keepdim=True)
        return self.g * (x - m) / (s + self.eps) + self.b


class PosEnc(nn.Module):
    def __init__(self, d_model, mx_len=1024, dropout=0.3):
        super().__init__()
        self.dp = nn.Dropout(p=dropout)
        
        pe = torch.zeros(mx_len, d_model)
        pos = torch.arange(0, mx_len, dtype=torch.float).unsqueeze(1)
        dt = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(pos * dt)
        pe[:, 1::2] = torch.cos(pos * dt)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dp(x)


class MHA(nn.Module):
    def __init__(self, d_model, heads, dropout=0.3):
        super().__init__()
        self.d = d_model
        self.h = heads
        self.dk = d_model // heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dp = nn.Dropout(dropout)
        self.attn_w = None  
    
    def forward(self, x, mask=None, ret_attn=False):  
        b, sl, _ = x.size()
        
        q = self.wq(x).view(b, sl, self.h, self.dk).transpose(1, 2)
        k = self.wk(x).view(b, sl, self.h, self.dk).transpose(1, 2)
        v = self.wv(x).view(b, sl, self.h, self.dk).transpose(1, 2)
        
        sc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        
        if mask is not None:
            sc = sc.masked_fill(mask == 0, float('-inf'))
        
        aw = F.softmax(sc, dim=-1)
        if ret_attn:  
            self.attn_w = aw.detach()
        aw = self.dp(aw)
        ao = torch.matmul(aw, v)
        
        ao = ao.transpose(1, 2).contiguous().view(b, sl, self.d)
        out = self.wo(ao)
        
        if ret_attn:  
            return out, self.attn_w
        return out

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.3):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.dp = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.l2(self.dp(F.relu(self.l1(x))))


class DecL(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.3):
        super().__init__()
        self.sa = MHA(d_model, num_heads, dropout)
        self.ff = FFN(d_model, d_ff, dropout)
        self.n1 = LN(d_model)
        self.n2 = LN(d_model)
        self.dp = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, ret_attn=False):  
        if ret_attn:  
            attn_out, attn_w = self.sa(self.n1(x), mask, ret_attn=True)
            x = x + self.dp(attn_out)
            x = x + self.dp(self.ff(self.n2(x)))
            return x, attn_w
        else:
            x = x + self.dp(self.sa(self.n1(x), mask))
            x = x + self.dp(self.ff(self.n2(x)))
            return x

class Trans(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_l=6, n_heads=8, 
                 d_ff=2048, mx_sq_len=1024, dropout=0.3, pad_idx=0):
        super().__init__()
        
        self.dm = d_model
        self.msl = mx_sq_len
        self.pi = pad_idx
        self.vs = vocab_size
        
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pe = PosEnc(d_model, mx_sq_len, dropout)
        
        self.lyrs = nn.ModuleList([
            DecL(d_model, n_heads, d_ff, dropout)
            for _ in range(n_l)
        ])
        
        self.fn = LN(d_model)
        self.out = nn.Linear(d_model, vocab_size)
        self.init_w()
        
        np = sum(p.numel() for p in self.parameters())
        print("Total " + format(np, ',') + " parameters")
    
    def init_w(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def ld_pre_emb(self, embedding_matrix, freeze=False):
        self.emb.weight.data = torch.FloatTensor(embedding_matrix)
        if freeze:
            self.emb.weight.requires_grad = False

    def forward(self, x, mask=None, ret_attn=False):  
        sl = x.size(1)
        
        if mask is None:
            mask = self.gen_mask(sl, x.device)
        
        x = self.emb(x)
        x = x * math.sqrt(self.dm)
        x = self.pe(x)
        
        attn_ws = []  
        for lyr in self.lyrs:
            if ret_attn:  
                x, attn_w = lyr(x, mask, ret_attn=True)
                attn_ws.append(attn_w)
            else:
                x = lyr(x, mask)
        
        x = self.fn(x)
        lg = self.out(x)
        
        if ret_attn:  
            return lg, attn_ws
        return lg
    
    def gen_mask(self, seq_len, device):
        msk = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return msk.unsqueeze(0).unsqueeze(0)
    
    def generate(self, prompt, max_new_tokens=100, temperature=1.0, top_k=50, eos_token_id=None):
        self.eval()
        gen = prompt.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if gen.size(1) > self.msl:
                    inp = gen[:, -self.msl:]
                else:
                    inp = gen
                    
                lg = self(inp)[:, -1, :] / temperature

                if top_k > 0:
                    v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
                    lg[lg < v[:, [-1]]] = -float('Inf')
                
                prb = F.softmax(lg, dim=-1)
                nt = torch.multinomial(prb, num_samples=1)
                gen = torch.cat([gen, nt], dim=-1)
                
                if eos_token_id is not None and nt.item() == eos_token_id:
                    break
        
        return gen