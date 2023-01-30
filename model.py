from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

# --------Hyperparameters---------
batch_size = 48 # 64 Independent sequences to process in parallel
block_size = 160 # 256 Max context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = torch.device("mps")
eval_iters = 200
n_embd = 192 # 384
n_head = 4 # 6
n_layer = 6 # 6
dropout = 0.2
# --------------------------------

torch.manual_seed(1337)

class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape # (B,T,n_embd)
        k = self.key(x) # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        # Compute attention affinities
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,C) @ (B,C,T) --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # Weighted aggregation of the values
        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B,T,T) @ (B,T,head_size) --> (B,T,head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Each head attends to the entire block,
        # taking input of (B,T,n_embd) and producing out of (B,T,head_size)
        # head_size = n_embd // num_heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out) # Projection applied to output before residual connection
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity"""

    def __init__(self, layer_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(layer_size, layer_size*4), # (n_embd,n_embd*4)
            nn.ReLU(),
            # Projection applied before residual connection
            nn.Linear(layer_size*4, layer_size), # (n_embd*4,n_embd)
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Allows tokens to 'think' on the info derived from attention
        return self.net(x)

class Block(nn.Module):
    """ Transformer block, comprised of communication followed by computation"""

    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size) # 4 heads, each (B,T,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Residual connection added to output MultiHeadAttention
        x = x + self.ffwd(self.ln2(x)) # Residual connection added to output FeedForward
        return x

class TinyTransformer(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,n_embd)
        x = tok_emb + pos_emb # (B,T,n_embd)
        x = self.blocks(x) # (B,T,n_embd)
        x = self.ln_f(x) # (B,T,n_embd)
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
        idx_cond = 0
        for i in tqdm(range(max_new_tokens)):
            if i == 0:
                idx_cond = idx[:,-1:]
            elif block_size == idx_cond.size(dim=1):
                idx_cond = idx[:,-5:]
            else:
                idx_cond = torch.cat((idx_cond, idx[:,-1:]), dim=1)

            # Crop idx to last block_size tokens
            # idx_cond = idx[:, -block_size:]
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
    
