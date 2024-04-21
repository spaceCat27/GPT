import torch
from torch import nn
import torch.nn.functional as F
import math
import random

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_size, attention_size) -> None:
        super().__init__()
        self.key = nn.Linear(embedding_size, attention_size, bias=False)
        self.query = nn.Linear(embedding_size, attention_size, bias=False)
        self.value = nn.Linear(embedding_size, attention_size, bias=False)
        self.embedding_size = embedding_size
        self.attention_size = attention_size

    def forward(self, embedding):
        k = self.key(embedding)
        q = self.key(embedding)
        v = self.key(embedding)
        x = q @ torch.transpose(k, 1, 2)
        x = x / math.sqrt(self.attention_size)
        x = x.masked_fill(torch.tril(x) == 0, float('-inf'))
        x = F.softmax(x, dim=2)
        x = x @ v
        
        return x