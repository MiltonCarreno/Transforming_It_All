import torch

# --------Hyperparameters---------
batch_size = 64 # 64 Independent sequences to process in parallel
block_size = 192 # 256 Max context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = torch.device("mps")
eval_iters = 200
n_embd = 240 # 384
n_head = 4 # 6
n_layer = 6 # 6
dropout = 0.2
# --------------------------------