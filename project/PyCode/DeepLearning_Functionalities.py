import numpy as np
import scipy as scp
import math as ma
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import collections
import sys
import os.path
import importlib as imp
import json
from scipy.optimize import curve_fit

import sklearn.cluster as skcltr
import sklearn.linear_model as skl_linear

import torch
import torch.nn as nn
import torch.nn.functional as nF
from torch import tensor
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def clone_state_dict_from(X):
    return collections.OrderedDict(
        [(key, tensor.clone().detach()) for key, tensor in X.state_dict().items()]
    )

class myDataset_from_tensors(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.f = x
        self.t = y
    def __len__(self):
        return len(self.f)
    def __getitem__(self, idx):
        sample = (self.f[idx], self.t[idx])
        return sample

class NeuralNet_MLP_Arch(nn.Module):
    def __init__(self, arch:list, act_fn = nn.LeakyReLU(negative_slope=0.1)):
        super(NeuralNet_MLP_Arch, self).__init__()
        if len(arch)<3:
            print('architecture wrong')
            sys.exit()
        self.architecture = arch
        self.activation = act_fn
        self.num_tot_layers = len(arch)
        self.num_tot_passages = len(arch)-1
        self.passage_struct = []
        intrmdt_idx = 0
        for i in range(len(arch)-1):
            if i == 0:
                # code = 'self.in_lyr = nn.Linear({0:d},{1:d}, dtype=torch.float64)'.format(arch[i], arch[i+1])
                lname = 'in_lyr'
                self._modules[lname] = nn.Linear(arch[i], arch[i+1], dtype=torch.float64)
                self.passage_struct.append(lname)
            elif i == len(arch)-2:
                # code = 'self.out_lyr = nn.Linear({0:d},{1:d}, dtype=torch.float64)'.format(arch[i], arch[i+1])
                lname = 'out_lyr'
                self._modules[lname] = nn.Linear(arch[i], arch[i+1], dtype=torch.float64)
                self.passage_struct.append(lname)
            else:
                # code = 'self.h{0:d} = nn.Linear({1:d},{2:d}, dtype=torch.float64)'.format(intrmdt_idx, arch[i], arch[i+1])
                lname = 'h{0:d}'.format(intrmdt_idx)
                self._modules[lname] = nn.Linear(arch[i], arch[i+1], dtype=torch.float64)
                intrmdt_idx = intrmdt_idx+1
                self.passage_struct.append(lname)
            # exec(code)
        
        self.num_intrmdt_passages = intrmdt_idx

    def forward(self, X):
        for i, lname in enumerate(self.passage_struct):
            if i == 0:
                out = self._modules[lname](X)
                out = self.activation(out)
            elif i == self.num_tot_passages-1:
                out = self._modules[lname](out)
            else:
                out = self._modules[lname](out)
                out = self.activation(out)
        return out

    def get_ave_grad(self):
        with torch.no_grad():
            num_list = []
            mean_list = []
            for i, lname in enumerate(self.passage_struct):
                for key in self._modules[lname]._parameters.keys():
                    mean_list.append(self._modules[lname]._parameters[key].detach().abs().mean())
                    num_elements = 1
                    for number in self._modules[lname]._parameters[key].shape:
                        num_elements = num_elements*number
                    num_list.append(num_elements)
            num_list = np.array(num_list)
            mean_list = np.array(mean_list)

            return np.sum(mean_list*num_list)/np.sum(num_list)

    def rand_refresh(self):
        for key in self.state_dict().keys():
            nn.init.uniform_(self.state_dict()[key], -1.0, 1.0)

    def dump_state_dict(self):
        return clone_state_dict_from(self)
