from multi_head_attention import MultiHeadAttention
import torch
from torch import nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, model_size, head_number) -> None:
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(model_size, head_number)
        self.norm_layer1 = nn.LayerNorm(model_size)
        self.norm_layer2 = nn.LayerNorm(model_size)
        self.linear_layer1 = nn.Linear(model_size, model_size)
        self.linear_layer2 = nn.Linear(model_size, model_size)

    def forward(self, embedding):
        x = self.multi_head_attention(embedding)
        x = self.norm_layer1(x) + embedding
        x2 = self.linear_layer1(x)
        x2 = F.relu(x2)
        x2 = self.linear_layer2(x2)
        x = self.norm_layer2(x2) + x

        return x

