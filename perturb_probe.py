'''
probe dataset里的一个sample, 求epsilon.
'''
import os
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
plt.rc('text', usetex=True)
plt.rc('font', **{'size': 14.0})
plt.rc('text.latex', preamble=r'\usepackage{lmodern}')
import sys
from Marabou.maraboupy import Marabou

mid_dim = 128
how_many_history_step_to_use = 99
exp = f"state_tl{mid_dim}"

'''
load a game from the intervention benchmark and select an intervention configuration
'''
with open("intervention_benchmark.pkl", "rb") as input_file:
    dataset = pickle.load(input_file)
# print("dataset lenngth: ", len(dataset)) # 1001
# print(dataset[777])
    
# one sample
case_id = 777 #{'history': [37, 29, 18, 42, 19], 'pos_int': 37, 'ori_color': 2.0}
completion = dataset[case_id]["history"]

# load nonlinear probe (state_tl128_layer1):
path = "./ckpts/battery_othello/state_tl128/layer1/checkpoint.onnx"
net1 = Marabou.read_onnx(path)


# TODO: get the hidden state and the preintervention label for the last time stamp

'''
load trained models
'''
othello = get_othello(ood_perc=0., data_root=None, wthor=False, ood_num=1) # OSError: [Errno 12] Cannot allocate memory
train_dataset = CharDataset(othello)
s
mconf = GPTConfig(61, 59, n_layer=8, n_head=8, n_embd=512)

models = {}
for layer in range(layer_s, layer_e):
    model = GPTforProbeIA(mconf, probe_layer=layer)
    # model = GPT(mconf)
    load_res = model.load_state_dict(torch.load("./ckpts/gpt_synthetic.ckpt" if not championship else "./ckpts/gpt_championship.ckpt"))
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model = model.to(device)
    _ = model.eval()
    models[layer] = model

'''
check the partial game progeression
'''
ab = OthelloBoardState()
ab.update(completion, prt=True)











# # get input and output variable numbers; [0] since first dimension is batch size
# inputVars = net1.inputVars[0] #input shape: [1024, 512]
# outputVars = net1.outputVars[0] #output shape: [1024, 64]
# epsilon = 0.03
# for x in range(inputVars.shape[0]): #512
#     net1.setLowerBound(inuptVars[])