import re
from tqdm import tqdm
import pickle
from pathlib import Path
import os
os.environ["HTTP_PROXY"] = "http://EEZ258127:your_password@10.10.78.61:3128"
os.environ["HTTPS_PROXY"] = "http://EEZ258127:your_password@10.10.78.61:3128"


class tinyPreprocessor:
    def __init__(self):
        self.stats = {
            'original_count': 0, 'after_cleaning': 0, 'too_short': 0,
            'too_long': 0, 'duplicates': 0, 'final_count': 0
        }
     
    def clean_text(self, text):
        text = re.sub(r'http\S+|www\.\S+|\S+@\S+|<.*?>|\bRT\b', '', text, flags=re.IGNORECASE)
        text = text.lower()
        text = re.sub(r"[^\w\s.,!?\'-]", " ", text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'([.,!?])([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'\.{2,}', '.', text)
        return text.strip()
    
    def preprocess(self, dataset):
        self.stats['original_count'] = len(dataset)
        
        preprocessed_texts = []
        seen_texts = set()
        
        for item in tqdm(dataset, desc='Processing'):
            text = self.clean_text(item['text'])
            
            if not text or len(text.split()) < 10 or len(text.split()) > 1000:
                continue
            
            if text in seen_texts:
                self.stats['duplicates'] += 1
                continue
            seen_texts.add(text)
            
            preprocessed_texts.append(text)
        
        self.stats['final_count'] = len(preprocessed_texts)
        return preprocessed_texts
    
    def print_stats(self):
        print(f"Original: {self.stats['original_count']:,}")
        print(f"Duplicates removed: {self.stats['duplicates']:,}")
        print(f"Final texts: {self.stats['final_count']:,}")
        retention = (self.stats['final_count'] / self.stats['original_count'] * 100)
        print(f"Retention: {retention:.1f}%")
    
    def save(self, texts, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(texts, f)


def main():
    from datasets import load_dataset
    tiny = load_dataset("roneneldan/TinyStories")
    print(f"Train samples: {len(tiny['train']):,}")
    print(f"Validation samples: {len(tiny['validation']):,}")

    train_proc = tinyPreprocessor()
    print(f'Preprocessing training samples')
    train_texts = train_proc.preprocess(tiny['train'])
    train_proc.print_stats()
    train_proc.save(train_texts, 'Dataset/train_preprocessed.pkl')

    val_split = 'validation' if 'validation' in tiny else 'train'
    val_proc = tinyPreprocessor()
    print(f'Preprocessing validation samples')
    val_texts = val_proc.preprocess(tiny[val_split])
    val_proc.print_stats()
    val_proc.save(val_texts, 'Dataset/val_preprocessed.pkl')

if __name__ == "__main__":
    main()