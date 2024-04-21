import torch
from torch import nn
from single_head_attention import SingleHeadAttention



class MultiHeadAttention(nn.Module):
    def __init__(self, model_size, head_number) -> None:
        super().__init__()
        self.attention_size = model_size // head_number
        self.heads = nn.ModuleList()
        for _ in range(head_number):
            self.heads.append(SingleHeadAttention(model_size, self.attention_size))

    def forward(self, embedding):
        x_list = []
        for head in self.heads:
            x = head(embedding)
            x_list += [x]
        output = torch.cat(x_list, dim=2)

        return output