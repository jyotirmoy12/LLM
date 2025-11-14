import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gensim.downloader as api
import torch.nn as nn
from collections import Counter

import os
os.environ["HTTP_PROXY"] = "http://EEZ258127:your_password@10.10.78.61:3128"
os.environ["HTTPS_PROXY"] = "http://EEZ258127:your_password@10.10.78.61:3128"


class Tokenizer:
    def __init__(self):
        self.w2i = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}  # word to index
        self.i2w = {v: k for k, v in self.w2i.items()}  # index to word
        self.mg = {}  # merges
        self.mp = {}  # merge priority

    def vocab(self, txts, vocab_size=30000):
        cnt = Counter()
        
        for txt in txts:
            for w in txt.split():
                cnt[w] += 1
        
        srt = sorted(cnt.items(), key=lambda x: x[1], reverse=True)  # sorted words
        
        idx = len(self.w2i)
        count = 0
        for w, f in srt:
            if count >= vocab_size - len(self.w2i):
                break

            if w not in self.w2i:
                self.w2i[w] = idx
                self.i2w[idx] = w
                idx += 1
                count += 1
    
    def train(self, txts, vocab_size=30000):
        vocab = {}
        for txt in txts:
            for w in txt.split():
                if w:
                    ch = ' '.join(list(w)) + ' </w>'
                    vocab[ch] = vocab.get(ch, 0) + 1
    
        n = 0  # num of merges
        mx = 30000  # maximum merges
        
        while n < mx:
            p = {}  # pairs
            for w, f in vocab.items():  # count pairs
                sym = w.split()
                for i in range(len(sym) - 1):
                    pr = (sym[i], sym[i + 1])
                    p[pr] = p.get(pr, 0) + f
            
            if not p:
                break
               
            bst = max(p, key=p.get)  # choosing maximum freq pair
            if p[bst] < 2:
                break  # if frequency <2 then break
                
            bg = ' '.join(bst)  # bigram
            rp = ''.join(bst)  # replacement
            new_vocab = {}
            # replace pair with new token
            for w in vocab:
                new_w = w.replace(bg, rp)
                new_vocab[new_w] = vocab[w]
            vocab = new_vocab
            
            self.mg[bst] = rp
            self.mp[bst] = n
            n += 1
            
            if n % 5000 == 0:
                print("merges: " + str(n) + "/" + str(mx))
        
        # final tokens
        tk = {}
        for w, f in vocab.items():
            for t in w.split():
                tk[t] = tk.get(t, 0) + f
        
        srt_tk = sorted(tk.items(), key=lambda x: x[1], reverse=True)
        sel = [t for t, _ in srt_tk[:vocab_size - len(self.w2i)]]
        
        idx = len(self.w2i)
        for t in sel:
            if t not in self.w2i:
                self.w2i[t] = idx
                self.i2w[idx] = t
                idx += 1
        
        print("Final vocabulary size: " + str(len(self.w2i)))
    
    def enc(self, txt):
        if not txt or not txt.strip():
            return [2, 3] 
       
        tokens = [2] 
        for w in txt.split():
            if w:
                w_tokens = self.apply_bpe(w)
                for x in w_tokens:
                    tokens.append(self.w2i.get(x, 1))  
        tokens.append(3) 
        return tokens
    
    def apply_bpe(self, w):
        sym = list(w) + ['</w>']
        
        while True:
            prs = [(sym[i], sym[i + 1]) for i in range(len(sym) - 1)]
            bst_p = None
            bst_priority = float('inf')

            for pr in prs:
                if pr in self.mp and self.mp[pr] < bst_priority:
                    bst_priority = self.mp[pr]
                    bst_p = pr
            
            if not bst_p:
                break
                
            new_sym = []
            i = 0
            while i < len(sym):
                if i < len(sym) - 1 and sym[i] == bst_p[0] and sym[i + 1] == bst_p[1]:
                    new_sym.append(bst_p[0] + bst_p[1])
                    i += 2
                else:
                    new_sym.append(sym[i])
                    i += 1
            sym = new_sym
        
        return sym
    
    def dec(self, indices):
        tokens = []
        for i in indices:
            if i not in [0, 2, 3]: 
                t = self.i2w.get(i, '<UNK>')
                tokens.append(t)
        txt = ''.join(tokens).replace('</w>', ' ')
        return txt.strip()
    
    def keep(self, pth):
        Path(pth).parent.mkdir(parents=True, exist_ok=True)
        d = {
            'word_to_id': self.w2i,
            'id_to_word': self.i2w,
            'merges': {str(k): v for k, v in self.mg.items()},
            'merge_priority': {str(k): v for k, v in self.mp.items()}
        }
        with open(pth, 'wb') as fp:
            pickle.dump(d, fp)
            
    @classmethod 
    def ld(cls, pth):
        with open(pth, 'rb') as fp:
            d = pickle.load(fp)
        tokenizer = cls()
        tokenizer.w2i = d['word_to_id']
        tokenizer.i2w = d['id_to_word']
        tokenizer.mg = {eval(k): v for k, v in d['merges'].items()}
        tokenizer.mp = {eval(k): v for k, v in d['merge_priority'].items()}
        return tokenizer


def ld_emb(tokenizer):
    # loading pretrained embeddings
    model = api.load('fasttext-wiki-news-subwords-300')
    
    vocab_size = len(tokenizer.w2i)
    emb = np.zeros((vocab_size, 300), dtype=np.float32)
    
    for t, idx in tqdm(tokenizer.w2i.items()):
        if t in ['<PAD>', '<UNK>', '<SOS>', '<EOS>']:
            if t == '<PAD>':
                emb[idx] = np.zeros(300)
            else:
                emb[idx] = np.random.randn(300) * 0.01
            continue
    
        clean = t.replace('</w>', '')
        
        try:
            emb[idx] = model.get_vector(clean)
            continue
        except KeyError:
            pass

        vecs = []
        for i in range(len(clean)):
            for j in range(i + 1, min(i + 4, len(clean) + 1)):
                try:
                    vec = model.get_vector(clean[i:j])
                    vecs.append(vec)
                except KeyError:
                    continue
        
        if vecs:
            emb[idx] = np.mean(vecs, axis=0)
        else:
            emb[idx] = np.random.randn(300) * 0.01
    
    return emb


def main():
    with open('Dataset/train_preprocessed.pkl', 'rb') as fp:
        trd = pickle.load(fp)  # train data
    with open('Dataset/val_preprocessed.pkl', 'rb') as fp:
        vald = pickle.load(fp)  # val data

    trt = trd  # train text
    valt = vald  # val text
  
    tokenizer = Tokenizer()
    tokenizer.train(txts=trt, vocab_size=30000)
    tokenizer.keep('Dataset/tokenizer.pkl')
    
    train_ids = []
    for txt in tqdm(trt, desc="Tokenizing train"):
        tokens = tokenizer.enc(txt)
        if len(tokens) > 2:  
            train_ids.append(tokens)

    val_ids = []
    for txt in tqdm(valt, desc="Tokenizing val"):
        tokens = tokenizer.enc(txt)
        if len(tokens) > 2:
            val_ids.append(tokens)

    with open('Dataset/train_tokenized.pkl', 'wb') as fp:
        pickle.dump(train_ids, fp)
    with open('Dataset/val_tokenized.pkl', 'wb') as fp:
        pickle.dump(val_ids, fp)
    
    emb = ld_emb(tokenizer)
    
    with open('Dataset/embeddings.pkl', 'wb') as fp:
        pickle.dump({
            'embeddings': emb,
            'vocab_size': len(tokenizer.w2i),
            'embedding_dim': emb.shape[1]
        }, fp)

if __name__ == "__main__":
    main()