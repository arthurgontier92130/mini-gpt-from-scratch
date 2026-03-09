import torch
import torch.nn as nn
from MHA import MultiHeadAttention
import math

class TransformerBlock(nn.Module):
    def __init__(self, D, N_h):
        super().__init__()
        self.D = D
        self.H = N_h
        self.ln1 = nn.LayerNorm(D)
        self.MHA = MultiHeadAttention(D, N_h)
        self.ln2 = nn.LayerNorm(D)
        self.FFN = nn.Sequential(
            nn.Linear(D, 4*D),
            nn.GELU(),
            nn.Linear(4*D, D)
        )
        
    def forward(self, x):
        x = self.MHA(self.ln1(x)) + x
        x = self.FFN(self.ln2(x)) + x
        
        return x