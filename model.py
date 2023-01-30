from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

# --------Hyperparameters---------
batch_size = 4 # 64 Independent sequences to process in parallel
block_size = 8 # 256 Max context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = torch.device("mps")
eval_iters = 200
n_embd = 32 # 384
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

class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape # (B,T,n_embd)
        k = self.key(x) # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        # Compute attention affinities
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,C) @ (B,C,T) --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        # Weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) --> (B,T,C)
        return out

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,n_embd)
        x = tok_emb + pos_emb # (B,T,n_embd)
        x = self.sa_head(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in tqdm(range(max_new_tokens)):
            # Crop idx to last block_size tokens
            idx_cond = idx[:, -block_size:]
            # Get predictions
            logits, loss = self(idx_cond)
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
for iter in tqdm(range(max_iters)):
    # Evaluate loss of model every 'eval_interval'
    if iter % eval_interval == 0 or iter == (max_iters - 1):
        losses = estimate_loss()
        print("\n-----")
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
