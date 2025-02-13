# make deterministic
from mingpt.utils import set_seed
set_seed(44)

import os
import math
import time
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import torch
from torch import distributed as dist
import torch.nn as nn
from torch.nn import functional as F
from data import get_othello
from data.othello import permit, start_hands, OthelloBoardState, permit_reverse
from mingpt.dataset import CharDataset
from mingpt.utils import sample
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
import os
import re

# Handle distributed training with SRUN
nodelist = re.findall(r"gpunode\[(\d+)-\d+\]|gpunode(\d+)", os.environ['SLURM_JOB_NODELIST'])
print("node list: ", nodelist)
if not nodelist or len(nodelist) > 1:
    raise ValueError(f"Cannot parse node list\nSLURM_JOB_NODELIST: {os.environ['SLURM_JOB_NODELIST']}")
nodelist = nodelist[0]
if nodelist[0]:
    os.environ['MASTER_ADDR'] = f"gpunode{nodelist[0]}"
else:
    os.environ['MASTER_ADDR'] = f"gpunode{nodelist[1]}"
os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
os.environ['NODE_RANK'] = os.environ['SLURM_NODEID']
os.environ['RANK'] = os.environ['SLURM_GTIDS']
# FIXME: to see if SLURM_NTASKS is the right envvar to use
os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']

print(torch.cuda.get_device_name())

dist.init_process_group()


'''load dataset'''
synthetic_or_championship = True
othello = get_othello(ood_num=-1, data_root=None if synthetic_or_championship else "data/othello_championship", wthor=True)
train_dataset = CharDataset(othello)
'''load model'''
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
model = GPT(mconf)

total_params = sum(p.numel() for p in model.parameters())

print("There are ", total_params, "parameters in the OthelloGPT") #24.13916015625 M  (25311744 parameters)
for name, param in model.named_parameters():
    print(f"Parameter: {name}, dtype: {param.dtype}") #dtype: torch.float32


'''
1. For storing the model and its states for training
------------------------------------------------------------
~25M params with FP32 precision
(4 + 4 + 12)* 25M = 500MB ~ 0.5GB (copy of params, copy of gradients; 12B for optimizer states)

2. For storing the activations:
------------------------------------------------------------
num_layer L = 8
precision p = 4
sequence_len s = 60
batch_size b = 8 * 512 / world_size = 1024
hidden_dim h = 512
attention_head_num a = 8

Therefore:

sqrt(Lpsbh(16 + 2/p + 2as/h + as/ph)) = 8 * 4 * 60 * 1024 * 512 * (16 + 0.5 + 1.875 + 0.234375) = 18.609375 * 32 * 60 * 1024 * 512 = 18,732,810,240B ~ 19GB -> sqrt ~ 4.3GB
'''