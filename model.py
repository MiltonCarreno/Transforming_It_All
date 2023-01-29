import torch
import torch.nn as nn
from torch.nn import functional as F

# --------Hyperparameters---------
batch_size = 48 # 64 Independent sequences to process in parallel
block_size = 96 # 256 Max context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = torch.device("mps")
eval_iters = 200
n_embd = 192 # 384
n_head = 4 # 6
n_layer = 6 # 6
dropout = 0.2
# --------------------------------

torch.manual_seed(1337)

# Get preprocessed dataset
with open("texts/tiny_borges.txt", "r", encoding="utf-8") as fd:
    text = fd.read()

# Get list of unique characters in dataset
chars = sorted(list(set(text)))
vocab_size = len(chars) # Set vocab_size
# print(f"Vocab Size: {vocab_size}\n{chars}")

# Encode and decoder for bidirectional character-integer translation
stoi = {c:i for i, c in enumerate(chars)}
itos = {i:c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Split data 90/10 for training/validation
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Function to get a batch of data from train or val data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1]for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Function to calculate the loss of model during training
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        
        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Get predictions
            logits, loss = self(idx)
            # Get predictions for the last char for all batches
            logits = logits[:,-1,:]
            # Get probabilities with softmax
            probs = F.softmax(logits, dim=-1)
            # Sample according to the probabilities
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sample
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
model = TinyTransformer()
m = model.to(device) 

# Optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    # Evaluate loss of model every 'eval_interval'
    if iter % eval_interval == 0 or iter == (max_iters - 1):
        losses = estimate_loss()
        print("-----")
        print(f"Step {iter}\nTrain Loss: {losses['train']:.4f}; Val Loss: {losses['val']:.4f}")
        
    xb, yb = get_batch('train') # Get batch of data
    logits, loss = m(xb, yb) # Get prediction
    optimizer.zero_grad(set_to_none=True) # Zero previous gradient values
    loss.backward() # Calculate new gradients for all parameters
    optimizer.step() # Update parameters according to gradient

# Generate from model
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\n\n********Predictions********")
print(decode(m.generate(idx, max_new_tokens=500)[0].tolist()))
print("***************************\n")
