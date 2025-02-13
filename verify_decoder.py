import os

# make deterministic
from mingpt.utils import set_seed
set_seed(44)
import math
import time
import numpy as np
from copy import deepcopy
import pickle
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
from torch.utils.data import Subset
from tqdm import tqdm
from matplotlib import pyplot as plt

from data import get_othello, plot_probs, plot_mentals
from data.othello import permit, start_hands, OthelloBoardState, permit_reverse
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig, GPTforProbeIA
from mingpt.utils import sample, intervene, print_board
from mingpt.probe_model import BatteryProbeClassification, BatteryProbeClassificationTwoLayer
championship = False
mid_dim = 128
how_many_history_step_to_use = 99
exp = f"state_tl{mid_dim}"
if championship:
    exp += "_championship"


'''load benchmark dataset and its samples'''
with open("intervention_benchmark.pkl", "rb") as input_file:
    dataset = pickle.load(input_file)
completions = [data["history"] for data in dataset]
print("-----------------benchmark samples are loaded-------------------")

'''load non-linear probes'''
probes = {}
layer_s = 1
layer_e = 9
for layer in range(layer_s, layer_e):
    p = BatteryProbeClassificationTwoLayer(torch.cuda.current_device(), probe_class=3, num_task=64, mid_dim=mid_dim)
    load_res = p.load_state_dict(torch.load(f"./ckpts/battery_othello/{exp}/layer{layer}/checkpoint.ckpt")) # state_tl128/layer[4:9]/checkpoint
    p.eval() # Set the module in evaluation mode.
    probes[layer] = p # probes用来装(layer数量, probe) pairs
print("-----------------non-linear probes are loaded-------------------")

'''load the decoder'''
mconf = GPTConfig(61, 59, n_layer=8, n_head=8, n_embd=512) #vocab_size = 61

# models = {}
# for layer in range(layer_s, layer_e):
model = GPTforProbeIA(mconf, probe_layer=-1) #probe_layer is the last layer
# model = GPT(mconf)

'''load the retrained model (no layer norm in the decoder)'''
retrained_path = "./ckpts/no_linear_norm/non_layernorm_gpt__at_20240810_100610.ckpt"
load_res = model.load_state_dict(torch.load(retrained_path))
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    model = model.to(device)
_ = model.eval()
    # models[layer] = model

'''
decoder:
  (ln_f): deleted.
  (head): Linear(in_features=512, out_features=61, bias=False)
'''
# print(model)
decoder = model.head
device = torch.cuda.current_device()
decoder = decoder.to(device)
_ = decoder.eval()
print("-----------------decoder is loaded-------------------")

'''load the decoder_zeta_dataset for measuring accuracy and robustness'''
import json
path = "./decoder_zeta_dataset.json"
with open(path, 'r') as file:
    json_data = file.read()
data = json.loads(json_data)
samples = data['layer_8']



# trial on case 777
x = samples[777]['x']
x = torch.tensor(x).to(device)
print('x shape: ', torch.tensor(x).shape)
y_cap = samples[777]['y_cap']
print('y_cap shape: ', torch.tensor(y_cap).shape)
y = samples[777]['y']
print("y: ", y)
pre_intv_pred = decoder(x)
print("pre_intv_pred shape: ", pre_intv_pred.shape)  #[1, 61]
print("pre_intv_pred: ", pre_intv_pred)