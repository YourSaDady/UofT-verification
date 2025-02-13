import numpy as np
# from matplotlib import pyplot as plt
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from data import get_othello, plot_probs, plot_mentals
from data.othello import permit, start_hands, OthelloBoardState, permit_reverse
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig, GPTforProbeIA

championship = False

mconf = GPTConfig(61, 59, n_layer=8, n_head=8, n_embd=512)

# models = {}
# for layer in range(layer_s, layer_e):
model = GPTforProbeIA(mconf, probe_layer=-1) #probe_layer is the last layer
# model = GPT(mconf)
retrained_path = "/u/momoka/othello_world/ckpts/gpt_synthetic.ckpt"
# retrained_path = "./ckpts/no_linear_norm/non_layernorm_gpt__at_20240815_091352.ckpt"
load_res = model.load_state_dict(torch.load(retrained_path))
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    model = model.to(device)
_ = model.eval()
    # models[layer] = model

'''
decoder:
  (ln_f): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (head): Linear(in_features=512, out_features=61, bias=False)
'''
print(model.head)
decoder = model.head.to('cuda')
print("here?")
dummy_input = torch.randn(1,1,512, device='cuda') #[1, T=1, F=512]
onnx_path = "./ckpts/decoder_original.onnx"
torch.onnx.export(
    decoder, 
    dummy_input, 
    onnx_path, 
    input_names = ['input'],
    dynamic_axes={'input': {1: 'time_stamp'}}
)  # the second dimension T should be flexible for any time stamp (number o f tiles played)
print("here??")