import os
import time
from tqdm import tqdm
import numpy as np
# from matplotlib import pyplot as plt
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from data import get_othello
from data.othello import permit, start_hands, OthelloBoardState
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig, GPTforProbing
from mingpt.probe_trainer import Trainer, TrainerConfig
from mingpt.probe_model import BatteryProbeClassification, BatteryProbeClassificationTwoLayer

exp = "state_tl128"

# # a random probe positional input (我也不知道为什么要有这个)
# p_input = torch.rand((1024, 512)).to('cuda:0')

# for layer in range(9):
#     p = BatteryProbeClassificationTwoLayer(torch.cuda.current_device(), probe_class=3, num_task=64, mid_dim=128)
#     load_res = p.load_state_dict(torch.load(f"./ckpts/battery_othello/{exp}/layer0/checkpoint.ckpt"))
#     p = p.to('cuda:0')
#     p.eval()
    
#     # export probe to ONNX format
#     output_path = f"./ckpts/battery_othello/{exp}/layer{layer}/checkpoint.onnx"
#     onnx_p = torch.onnx.export(p, p_input, output_path)
#     # onnx_p = torch.onnx.dynamo_export(p, p_input)
#     # onnx_p.save(f"./ckpts/battery_othello/{exp}/layer{layer}/checkpoint.onnx")

#     # # export probe to NNet format
#     # output_path =  f"./ckpts/battery_othello/{exp}/layer{layer}/checkpoint.nnet"
#     # torch.save(p.state_dict(),output_path)

# # from maraboupy import Marabou
# # import numpy as np
# # # Set the Marabou option to restrict printing
# # options = Marabou.createOptions(verbosity = 0)
# # # Fully-connected network example
# # # -------------------------------
# # #
# # # This network has inputs x0, x1, and was trained to create outputs that approximate
# # # y0 = abs(x0) + abs(x1), y1 = x0^2 + x1^2
# # print("Fully Connected Network Example")
# # filename = "../Marabou/resources/onnx/fc1.onnx"
# # network = Marabou.read_onnx(filename)
# # # Get the input and output variable numbers; [0] since first dimension is batch size
# # inputVars = network.inputVars[0][0]
# # outputVars = network.outputVars[0][0]
# # print("network.inputVars[0][0]: ", inputVars)
# # print("network.outputVars[0][0]: ", outputVars)

import sys
from Marabou.maraboupy import Marabou
from Marabou.maraboupy.MarabouCore import StatisticsUnsignedAttribute

# 
# Path to NNet file
nnetFile = "../../src/input_parsers/acas_example/ACASXU_run2a_1_1_tiny_2.nnet"

# 
# Load the network from NNet file, and set a lower bound on first output variable
net1 = Marabou.read_nnet(nnetFile) # Constructs a MarabouNetworkNnet object from a .nnet file
net1.setLowerBound(net1.outputVars[0][0][0], .5)

# 
# Solve Marabou query
exitCode, vals1, stats1 = net1.solve()


# 
# Example statistics
stats1.getUnsignedAttribute(StatisticsUnsignedAttribute.NUM_SPLITS)
stats1.getTotalTimeInMicro()


#
# Eval example
#
# Test that when the upper/lower bounds of input variables are fixed at the
# same value, with no other input/output constraints, Marabou returns the 
# outputs found by evaluating the network at the input point.
inputs = np.array([-0.328422874212265,
                    0.40932923555374146,
                   -0.017379289492964745,
                   -0.2747684121131897,
                   -0.30628132820129395])

outputsExpected = np.array([0.49999678, -0.18876659,  0.80778555, -2.76422264, -0.12984317])

net2 = Marabou.read_nnet(nnetFile)
outputsMarabou = net2.evaluateWithMarabou([inputs])
assert max(abs(outputsMarabou[0].flatten() - outputsExpected)) < 1e-8