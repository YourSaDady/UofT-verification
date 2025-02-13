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




# # importing os module  
# import os 
# import pprint 
  
# # Get the list of user's 
# env_var = os.environ 
  
# # Print the list of user's 
# print("User's Environment variable:") 
# pprint.pprint(dict(env_var), width = 1) 


# exit()








os.environ['MASTER_PORT'] = '12838'

print("\n\nos.environ['SLURM_JOB_NODELIST']: ",  os.environ['SLURM_JOB_NODELIST'], "\n\n")
match = re.findall(r'\d+', os.environ['SLURM_JOB_NODELIST'])
print("match: ", match)
os.environ['MASTER_ADDR'] = f"gpunode{match[0]}"

# # Handle distributed training with SRUN
# nodelist = re.findall(r"gpunode\[(\d+)-\d+\]|gpunode(\d+)", os.environ['SLURM_JOB_NODELIST'])
# print("node list: ", nodelist)
# if not nodelist or len(nodelist) > 1:
#     raise ValueError(f"Cannot parse node list\nSLURM_JOB_NODELIST: {os.environ['SLURM_JOB_NODELIST']}")
# nodelist = nodelist[0]
# if nodelist[0]:
#     os.environ['MASTER_ADDR'] = f"gpunode{nodelist[0]}"
# else:
#     os.environ['MASTER_ADDR'] = f"gpunode{nodelist[1]}"
os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
os.environ['NODE_RANK'] = os.environ['SLURM_NODEID']
os.environ['RANK'] = os.environ['SLURM_GTIDS']
# FIXME: to see if SLURM_NTASKS is the right envvar to use
os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']

print(torch.cuda.get_device_name())

dist.init_process_group()

# torch.cuda.set_device(dist.get_rank())
torch.cuda.empty_cache()

def printMemory():
    for i in range(torch.cuda.device_count()):
        print(f"DEVICE: {i}, torch.cuda.memory_allocated: {torch.cuda.memory_allocated(i)/1024/1024/1024}GB")
        print(f"DEVICE: {i}, torch.cuda.memory_reserved: {torch.cuda.memory_reserved(i)/1024/1024/1024}GB")
        print(f"DEVICE: {i}, torch.cuda.max_memory_reserved: {torch.cuda.max_memory_reserved(i)/1024/1024/1024}GB")

# Function to print memory usage
def print_memory_usage(message):
    allocated = torch.cuda.memory_allocated()
    max_allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_reserved = torch.cuda.max_memory_reserved()
    print(f"{message} - Allocated: {allocated}/{max_allocated}, Reserved: {reserved}/{max_reserved}")

# # enable memory history, which will
# # add tracebacks and event history to snapshots
# torch.cuda.memory._record_memory_history()

synthetic_or_championship = True

othello = get_othello(ood_num=-1, data_root=None if synthetic_or_championship else "data/othello_championship", wthor=True)
train_dataset = CharDataset(othello)

# print_memory_usage("\n\n--------------------------\nAfter loaded dataset - mem usage: ")
# print("-------------------------\n\n")

# print("syntheticdataset")

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
model = GPT(mconf)

# print_memory_usage("\n\n--------------------------\nAfter loaded model - mem usage: ")
# print("-------------------------\n\n")
# torch.cuda.memory._snapshot()

max_epochs = 250
# initialize a trainer instance and kick off training
t_start = time.strftime("_%Y%m%d_%H%M%S")
assert 512 % dist.get_world_size() == 0
tconf = TrainerConfig(
    max_epochs=max_epochs, 
    batch_size= 8*512 // dist.get_world_size(),  # assuming 8 GPU's
    learning_rate=5e-4, #original: 5e-4
    lr_decay=True, 
    warmup_tokens=len(train_dataset)*train_dataset.block_size*5, 
    final_tokens=len(train_dataset)*train_dataset.block_size*max_epochs,
    num_workers=0,  #?
    ckpt_path=f"./ckpts/no_linear_norm/non_layernorm_gpt__at{t_start}.ckpt", 

    # ./ckpts/no_linear_norm/non_layernorm_gpt__at_20240807_140144.ckpt

)
trainer = Trainer(model, train_dataset, None, tconf)
device = trainer.device

# print_memory_usage("\n\n--------------------------\nAfter loaded trainer - mem usage: ")
# print("-------------------------\n\n")
torch.cuda.memory._snapshot()



print(t_start)
# print("start training!")
# print_memory_usage("\n\n--------------------------\nmem usage: ")
# print("-------------------------\n\n")
# # torch.cuda.memory._dump_snapshot("./ckpts/before_train.pickle")
# print("\n\nmemory history saved!\n\n")
print("torch.cuda.device_count(): ", torch.cuda.device_count())
printMemory()
trainer.train(recover=True, previous_ckpt="./ckpts/no_linear_norm/non_layernorm_gpt__at_20240813_163144.ckpt")
print("training finished!")
# trainer.save_checkpoint() #在train的最后save过了
print('saving finished, save to ', trainer.config.ckpt_path)


# total_params = sum(p.numel() for p in model.parameters())
# print(f"The total number of parameters in the model: {total_params}")