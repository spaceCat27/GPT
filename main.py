import torch
import random
from gpt import GPT



with open("data/krawczyk.txt", "r", encoding="utf-8") as f:
    file_text = f.read()

file_text = file_text.lower()
characters = list(set(file_text))
characters_num = len(characters)

int_to_char = {i:c for i,c in enumerate(characters)}
char_to_int = {c:i for i,c in enumerate(characters)}

data = [char_to_int[file_text[i]] for i in range(len(file_text))]


def get_batch(batch_size, context_length):
    x = torch.zeros(batch_size, context_length, dtype=torch.int64)
    y = torch.zeros(batch_size, context_length, dtype=torch.int64)

    for batch_element in range(batch_size):
        ix = random.randint(0, len(file_text) - context_length - 1)
        for context in range(context_length):
            x[batch_element, context] = data[ix + context] 
            y[batch_element, context] = data[ix + context + 1] 

    return x, y


batch_size = 32
context_length = 32
emb_num = 64
epochs = 101
model = GPT(1, emb_num, 2, characters_num, context_length)
lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


for e in range(epochs):
    Xtr, ytr = get_batch(batch_size, context_length)

    loss = model.forward(Xtr, ytr)

    if e % 10 == 0:
        print(f'{e} loss: {loss}')

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


context = torch.tensor([[char_to_int['\n']]], dtype=torch.int64)
num_of_tokens = 1024
s = ''
for i in range(num_of_tokens):
    c = model.genereate(context[:, -context_length:])
    s += int_to_char[c]
    context = torch.cat((context, torch.tensor([[c]])), dim=1)


print(s)
