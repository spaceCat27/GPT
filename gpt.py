import torch
from torch import nn
import torch.nn.functional as F
import math
from transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(self, transformer_block_number, model_size, head_number, vocab_size, context_length) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, model_size)
        self.positional_embeddings = nn.Embedding(context_length, model_size)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(transformer_block_number):
            self.transformer_blocks.append(TransformerBlock(model_size, head_number))
        self.norm_layer = nn.LayerNorm(model_size)
        self.linear_layer = nn.Linear(model_size, vocab_size)



    def forward(self, context, Y):
        x = self.embeddings(context) + self.positional_embeddings(torch.arange(context.shape[1]))
        
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm_layer(x)
        x = self.linear_layer(x)
        logits = x.view(x.shape[0] * x.shape[1], x.shape[2])
        Y = Y.view(Y.shape[0] * Y.shape[1])
        loss = F.cross_entropy(logits, Y)

        return loss
    
    def genereate(self, context):
        x = self.embeddings(context) + self.positional_embeddings(torch.arange(context.shape[1]))
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm_layer(x)
        x = self.linear_layer(x)
        x = x.view(x.shape[0] * x.shape[1], x.shape[2])
        x = F.softmax(x, dim=-1)
        x = torch.multinomial(x[-1], 1).item()

        return x
