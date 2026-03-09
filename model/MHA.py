import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_emb, n_h):
        super().__init__()
        assert d_emb % n_h == 0
        
        self.d = d_emb
        self.n_h = n_h
        self.d_h = d_emb//n_h
        
        self.Wq = nn.Linear(self.d, self.d)
        self.Wk = nn.Linear(self.d, self.d)
        self.Wv = nn.Linear(self.d, self.d)
        self.Wo = nn.Linear(self.d, self.d)
    
    def forward(self, x):
        B, S, D = x.shape
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        
        Qh = Q.reshape(B, S, self.n_h, self.d_h).permute(0,2,1,3) ## (B, H, S, Dh)
        Kh = K.reshape(B, S, self.n_h, self.d_h).permute(0,2,1,3) ## (B, H, S, Dh)
        Vh = V.reshape(B, S, self.n_h, self.d_h).permute(0,2,1,3) ## (B, H, S, Dh)
        
        scores = (Qh @ Kh.transpose(-1, -2))/self.d_h**0.5 ## (B, H, S, S)
        mask = torch.tril(torch.ones(S,S))
        scores_masked = scores.masked_fill(mask == 0, float("-inf"))
        
        attn = torch.softmax(scores_masked, dim=-1) ## (B, H, S, S)
        
        output_brut = (attn @ Vh).permute(0, 2, 1, 3) ## (B, S, H, Dh)
        
        output = output_brut.reshape(B, S, D)
        
        return self.Wo(output)


        
        
        
        