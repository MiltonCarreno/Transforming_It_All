import torch

# --------Hyperparameters---------
batch_size = 16 # 64 Independent sequences to process in parallel -- 64
block_size = 96 # 256 Max context length for predictions -- 192
max_iters = 5000 # -- 5000
eval_interval = 500 # -- 500
learning_rate = 3e-4 # -- 3e-4
device = torch.device("mps")
eval_iters = 200 # -- 200
n_embd = 100 # 384 -- 240
n_head = 4 # 6 -- 4
n_layer = 6 # 6 -- 6
dropout = 0.2 # -- 0.2
# --------------------------------

#--------BEST MODEL HPs---------
# batch_size = 64
# block_size = 192
# max_iters = 5000
# eval_interval = 500
# learning_rate = 3e-4
# device = torch.device("mps")
# eval_iters = 200 
# n_embd = 240
# n_head = 4
# n_layer = 6
# dropout = 0.2
# --------------------------------