import sys
sys.path.append("../GOProFormer")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, dim_embed, dim_ff, dropout=0.3):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_embed, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))



# class TermsSeqRelationForward(nn.Module):
#     def __init__(self, batch_size, dim_embed, dropout=0.3) -> None:
#         super(TermsSeqRelationForward, self).__init__()
#         self.linear = nn.Linear(batch_size, dim_embed)

#     def forward(self, x):
#         return F.dropout(F.relu(self.linear(x)))



class MultiheadAttentionWrapper(nn.Module):
    """This class will be used to extend MultiheadAttention."""
    def __init__(self, dim_embed, n_attn_heads) -> None:
        super(MultiheadAttentionWrapper, self).__init__()
        self.attn = nn.MultiheadAttention(dim_embed, n_attn_heads, batch_first=True)
    
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        attn_output, attn_weights = self.attn(query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask, average_attn_weights=False)
        # print(attn_output.shape, attn_weights.shape)
        return attn_output, attn_weights


class Embeddings(nn.Module):
    def __init__(self, vocab_size, dim_embed):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab_size, dim_embed, padding_idx=0)
        self.dim_embed = dim_embed

    def forward(self, x):
        return self.embed(x) * np.sqrt(self.dim_embed)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, dim_embed, dropout=0.3, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim_embed)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_embed, 2) *
                             -(np.log(10000.0) / dim_embed))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].clone().detach().requires_grad_(False)
        return self.dropout(x)