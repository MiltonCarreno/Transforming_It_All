import torch
from tqdm import tqdm
from model import TinyTransformer
from hyperparams import (batch_size, block_size, max_iters, 
                         eval_interval, learning_rate, device,
                         eval_iters)

# Get preprocessed dataset
with open("texts/tiny_borges.txt", "r", encoding="utf-8") as fd:
    text = fd.read()

# Get list of unique characters in dataset
chars = sorted(list(set(text)))
vocab_size = len(chars) # Set vocab_size

# Encode and decode for bidirectional character-integer translation
stoi = {c:i for i, c in enumerate(chars)} # Dictionary chars to nums
itos = {i:c for i, c in enumerate(chars)} # Dictionary nums to chars
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # Get 4 (i.e. batch_size) random samples
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Get context for each random sample of size 8 (i.e. block_size)
    x = torch.stack([data[i:i+block_size] for i in ix])
    # Get target for each random sample of size 8 (i.e. block_size)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # Pass data to device
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

if __name__ == "__main__":
    torch.manual_seed(1337)

    # Split data 90/10 for training/validation
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    model = TinyTransformer(vocab_size)
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
        logits, loss = m(xb, yb) # Calculate predictions and loss
        optimizer.zero_grad(set_to_none=True) # Zero previous gradient values
        loss.backward() # Calculate new gradients for all parameters
        optimizer.step() # Update parameters according to new gradient

    torch.save(m.state_dict(), "saved_model.pt")