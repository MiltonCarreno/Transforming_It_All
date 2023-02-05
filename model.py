from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from hyperparams import (block_size, device, n_embd, 
                         n_head, n_layer, dropout)

torch.manual_seed(1337)

class Head(nn.Module):
    """ One head of attention; capable of masking, amd either self or cross-attention """

    def __init__(self, head_size, masking):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.masking = masking

    def forward(self, xd, xe):
        B,T,C = xd.shape # (B,T,n_embd)
        # Self-attention: 'k','q','v' come from decoder
        # Cross-attention: 'k' and 'v' come from encoder, 'q' from decoder
        k = self.key(xd) if xe is None else self.key(xe) # (B,T,head_size)
        q = self.query(xd) # (B,T,head_size)
        v = self.value(xd) if xe is None else self.value(xe) # (B,T,head_size)
        # Compute attention affinities
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,C) @ (B,C,T) --> (B,T,T)
        if self.masking: # Masking for decoder but not for encoder
            wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T) **Masking**
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # Weighted aggregation of the values
        out = wei @ v # (B,T,T) @ (B,T,head_size) --> (B,T,head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention """

    def __init__(self, num_heads, head_size, masking):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, masking) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xd, xe=None):
        # Each head attends to the entire block,
        # taking input (B,T,n_embd) and producing output (B,T,head_size)
        out = torch.cat([h(xd, xe) for h in self.heads], dim=-1)
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
        self.sa = MultiHeadAttention(n_heads, head_size, masking=True) # 4 heads, each (B,T,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Residual connection added to output MultiHeadAttention
        x = x + self.ffwd(self.ln2(x)) # Residual connection added to output FeedForward
        return x

class Encoder(nn.Module):
    """ Encoder block """

    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, masking=False) # 4 heads, each (B,T,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Residual connection added to output MultiHeadAttention
        x = x + self.ffwd(self.ln2(x)) # Residual connection added to output FeedForward
        return x
    
class Decoder(nn.Module):
    """ Decoder block """

    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.msa = MultiHeadAttention(n_heads, head_size, masking=True) # 4 heads, each (B,T,head_size)
        self.ca = MultiHeadAttention(n_heads, head_size, masking=False) # 4 heads, each (B,T,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.ln4 = nn.LayerNorm(n_embd)

    def forward(self, xd, xe):
        x = xd + self.msa(self.ln1(xd)) # Masked self-attention + residual connection
        x = x + self.ca(self.ln2(x), self.ln3(xe)) # Non-masked cross-attention + residual connection
        x = x + self.ffwd(self.ln4(x)) # Feed forward layer + residual connection
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
    
class FullTransformer(nn.Module):

    def __init__(self, encoder_vocab_size, decoder_vacab_size):
        super().__init__()
        self.encoder_token_embedding_table = nn.Embedding(encoder_vocab_size, n_embd)
        self.decoder_token_embedding_table = nn.Embedding(decoder_vacab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Encoder
        self.encoders = nn.Sequential(*[Encoder(n_embd, n_head) for _ in range(n_layer)])
        # Decoder
        self.decoders = nn.Sequential(*[Decoder(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm
        self.lm_head = nn.Linear(n_embd, decoder_vacab_size)

    def forward(self, encoder_input=None, decoder_input=None, training=True):
        B1, T1 = encoder_input.shape
        B2, T2 = decoder_input.shape

        pos_emb_x = self.position_embedding_table(torch.arange(T1, device=device)) # (T,n_embd)
        pos_emb_y = self.position_embedding_table(torch.arange(T2, device=device)) # (T,n_embd)

        tok_emb_x = self.encoder_token_embedding_table(encoder_input) # (B,T,n_embd)
        tok_emb_y = self.decoder_token_embedding_table(decoder_input) # (B,T,n_embd)
    
        x = tok_emb_x + pos_emb_x # (B,T,n_embd) Embedded input for encoder
        y = tok_emb_y + pos_emb_y # (B,T,n_embd) Embedded input for decoder

        encoder_output = self.encoders(x) # (B,T,n_embd)
        decoder_output = self.decoders(y, encoder_output) # (B,T,n_embd)

        out = self.ln_f(decoder_output) # (B,T,n_embd)
        logits = self.lm_head(out) # (B,T,vocab_size)

        if not training:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            decoder_input = decoder_input.view(B*T)
            loss = F.cross_entropy(logits, decoder_input)
        
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
    
