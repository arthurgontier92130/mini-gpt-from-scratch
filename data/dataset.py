"""
dataset.py - Dataset, Vocabulary, and DataLoaders
===================================================

GOAL:
    Build a character-level dataset for training the GPT model.
    This file handles: reading the text, building a vocabulary (char-to-index mapping),
    encoding/decoding text, and creating PyTorch DataLoaders.



"""

import torch
from torch.utils.data import Dataset, DataLoader
import os

DATA_DIR = "data"
FILE_NAME = "input.txt"
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

with open(FILE_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))

stoi = {}
itos = {}

i = 0
for c in chars:
    stoi[c] = i
    itos[i] = c
    i+=1

vocab_size = len(chars)

def encode(text):
    l = []
    for c in text:
        l.append(stoi[c])
    return l

def decode(tokens):
    s = []
    for i in tokens:
        s.append(itos[i])
    return "".join(s)

encoded_data = torch.tensor(encode(text), dtype=torch.long)

print(vocab_size)
print(encoded_data[:20])
print(decode(encoded_data[:20].tolist()))

split_idx = int(len(encoded_data)*0.9)

train_data = encoded_data[:split_idx]
val_data = encoded_data[split_idx:]

class ShakespeareDataset(Dataset):
    def __init__(self, data, block_size):
        super().__init__()
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size 
    
    def __getitem__(self, index):
        x = self.data[index:index+self.block_size]
        y = self.data[index+1:index+1+self.block_size]
        return (x,y)


def get_dataloader(block_size, batch_size):
    train_loader = DataLoader(
        dataset=ShakespeareDataset(data=train_data, block_size=block_size), 
        batch_size=batch_size
    )
    
    val_loader = DataLoader(
        dataset=ShakespeareDataset(data=val_data, block_size=block_size),
        batch_size=batch_size
    )
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, _ = get_dataloader(block_size=128, batch_size=32)
    for x, y in train_loader:
        print(f"Batch x shape:{x.shape}")
        print(f"Batch y shape:{y.shape}")
        print(f"Exemple input: {x[0][:10]}")
        print(f"Exemple input: {y[0][:10]}")
        break
