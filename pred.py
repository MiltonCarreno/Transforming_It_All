import torch
from train import (decode, vocab_size)
from model import TinyTransformer
from hyperparams import device

torch.manual_seed(1337)

model = TinyTransformer(vocab_size)
model.load_state_dict(torch.load("saved_model_3.pt"))
m = model.to(device)
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Print number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

m.eval()

a = m.generate(context, max_new_tokens=2000)[0].tolist()
print("\n\n********Predictions********")
print(decode(a))
print("***************************\n")

# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# a = decode(m.generate(context, max_new_tokens=10000)[0].tolist())
# open("more.txt", "w").write(a)

