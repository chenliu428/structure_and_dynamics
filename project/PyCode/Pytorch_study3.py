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


# for data loading
import data_read as dr 

# for cross-validation data preparation 
import data_prep_crossvalid as dxv

# handmade linear regressors
import simpleLinearReg as slr

# import functions from other training methods
import GaussianLH_Panelty_RidgeLasso_MAP as glh_map
import Pytorch_study2 as pt_study2

imp.reload(dr)
imp.reload(dxv)
imp.reload(slr)
imp.reload(glh_map)
imp.reload(pt_study2)

def dynamic_forloops(depth:int, loops_length:list, idx_ary:np.ndarray, dim:int, counter:list=[0]):
    # print(indexs)
    # print('lenloop:', len(loops_length), ' vs depth.', depth)
    if len(loops_length)!=depth: 
        print('wrong loop info, exit program')
        sys.exit()
    
    if depth==0:
        counter[0] = counter[0] + 1
        print ('print idx: ', idx_ary, ' cnt.', counter[0])
    else:
        for i in range(loops_length[0]):
            idx_ary[dim-depth] = i
            dynamic_forloops(depth-1, loops_length[1:], idx_ary, dim, counter)    

def im_new(x):
    plt.figure()
    plt.imshow(x)
    plt.colorbar()

def compare_plot(index:list, Y_progress, train_dset):
    clrs = cm.rainbow(np.linspace(0,1,len(index)))
    figp, axp = plt.subplots()
    for i, idx in enumerate(index):
        axp.plot(train_dset[:][1].detach(), Y_progress[idx], 'o', ms=7, mfc='none', color=clrs[i], alpha=0.6)
    return figp, axp

def Arch_String(arch:list):
    for i in range(len(arch)):
        if i==0:
            arch_str = f'In={arch[i]:d} -> '
        elif i==len(arch)-1:
            arch_str = arch_str + f'Out={arch[i]:d}'
        else:
            arch_str = arch_str + f'{arch[i]:d} -> '
    return arch_str

def animation_yvsy(Y_progress, Out_epochs, TrueTargets, sampling_period, pause_time=0.2):
    clrs = cm.rainbow(np.linspace(0,1,len(Out_epochs)))
    num_plots = int(len(Out_epochs)/sampling_period)
    plt.figure()
    for i in range(num_plots):
        idx = i*sampling_period
        plt.title(f'{idx:d} / {Out_epochs[-1]:d}')
        plt.plot(TrueTargets, Y_progress[0], 'o', ms=7, mfc='none', color=clrs[0], alpha=0.6)
        plt.plot(TrueTargets, Y_progress[-1], 'o', ms=7, mfc='none', color=clrs[-1], alpha=0.6)
        plt.plot(TrueTargets, Y_progress[idx], 'o', ms=7, mfc='none', color=clrs[idx], alpha=0.6)
        plt.pause(pause_time)
        plt.clf()

def person_coeffcicient(X:np.ndarray, Y:np.ndarray):
    X_bar = np.mean(X)
    Y_bar = np.mean(Y)
    XY_bar = np.mean(X*Y)
    return (XY_bar-X_bar*Y_bar)/(np.std(X)*np.std(Y))

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

class NeuralNet_MLP(nn.Module):
    def __init__(self, in_dim, h1, out_dim, act_fn = nn.ReLU, **keys):
        super(NeuralNet_MLP, self).__init__()
        self.activation = act_fn
        self.p_in = nn.Linear(in_dim, h1)
        self.p_out = nn.Linear(h1, out_dim)

    def forward(self, X):
        out = self.p_in(X)
        out = self.activation(out)
        out = self.p_out(out)
        return out

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

def Train_MLP_Arch(
        mymlp:NeuralNet_MLP_Arch, 
        loss, 
        nnOptmzr, 
        lr, 
        train_dset:myDataset_from_tensors, 
        valid_dset:myDataset_from_tensors, 
        train_dloader, 
        valid_dloader, 
        mav_smpl_size:int = 300, 
        num_epochs:int = 10000,
        out_period:int = 1,
        print_step:int = 200,
        tol_lag_r = 0.4,
    ):

    tol_lag = int(tol_lag_r*num_epochs)
    
    init_para = mymlp.dump_state_dict()

    optimizer = nnOptmzr(mymlp.parameters(), lr=lr)

    ## training ##
    Out_epochs = []
    Y_progress = []
    Err_train = []
    Err_valid = []
    Ave_Grad = []
    Err_train_mav = []
    Err_valid_mav = []
    
    ave_abs_grad = 1e20
    train_err = 1e10
    min_err_sofar = 1e10
    min_vlderr_sofar = 1e10
    epoch_lag = 1
    epoch_lag_vld = 1
    min_sofar_idx = 0
    min_sofar_idx_vld = 0
    mdlslct_sofar = mymlp.dump_state_dict()
    mdlslct_sofar_vld = mymlp.dump_state_dict()
    epoch = 0
    
    while epoch<num_epochs and epoch_lag_vld<tol_lag:
        for i, (f,t) in enumerate(train_dloader):

            yfwd = mymlp(f)
            err = loss(yfwd, t)
            # yfwd = mymlp(train_dset[:][0])
            # err = loss(yfwd, train_dset[:][1])
            
            err.backward()
            optimizer.step()
            optimizer.zero_grad()

            # if i%print_step==0:
            #     print('Epoch.{0:d}/{1:d}, Step.{2:d}/{3:d}, Loss.{4:.8f}'.format(epoch, num_epochs, i, num_steps, err.item()))

        Out_epochs.append(epoch)

        with torch.no_grad():
            y_inter = mymlp(train_dset[:][0]).detach()
            err_inter = loss(y_inter, train_dset[:][1])
            train_err = err_inter.item()
            Err_train.append(train_err)
            Y_progress.append(y_inter)

            y_itr_valid = mymlp(valid_dset[:][0])
            err_itr_valid = loss(y_itr_valid, valid_dset[:][1])
            valid_err = err_itr_valid.item()
            Err_valid.append(valid_err)

        if min_err_sofar>train_err:
            min_err_sofar = train_err
            min_sofar_idx = epoch
            mdlslct_sofar = mymlp.dump_state_dict()
        
        if min_vlderr_sofar>valid_err:
            min_vlderr_sofar = valid_err
            min_sofar_idx_vld = epoch
            mdlslct_sofar_vld = mymlp.dump_state_dict()
        
        epoch_lag = epoch - min_sofar_idx
        epoch_lag_vld = epoch - min_sofar_idx_vld

        si = max(0, epoch-mav_smpl_size)
        Err_train_mav.append(np.mean(np.array(Err_train[si:])))
        Err_valid_mav.append(np.mean(np.array(Err_valid[si:])))

        if epoch%print_step==0:
            print(f'Epoch.{epoch:d}/{num_epochs:d}, Loss.{train_err:.8E}, min_err. t.{min_err_sofar:.8E}, v.{min_vlderr_sofar:.8E}, lag.{epoch_lag_vld:d} /{tol_lag:d}')

        epoch = epoch + 1
    else:
        print(f'Epoch: [{epoch:d}/{num_epochs:d}], train_err.{train_err:.4E}, min_err_trn.{min_err_sofar:.4E}, min_err_vld.{min_vlderr_sofar:.4E}, lag. t.{epoch_lag:d} & v.{epoch_lag_vld:d} / {tol_lag:d}')

        print('restore the best model selections:')
        
        mymlp.load_state_dict(mdlslct_sofar)
        mymlp.eval()
        with torch.no_grad():
            yrstore = mymlp(train_dset[:][0])
            lrstore = loss(yrstore, train_dset[:][1])
        print(f'restored min train err: {lrstore.item():.8E}')
        
        mymlp.load_state_dict(mdlslct_sofar_vld)
        mymlp.eval()
        with torch.no_grad():
            yrstore_vld = mymlp(valid_dset[:][0])
            lrstore_vld = loss(yrstore_vld, valid_dset[:][1])
        print(f'restored min valid err: {lrstore_vld.item():.8E}')

        stop_state = 'reach_wall '*(epoch>=num_epochs) + '& vld_saturation'*(epoch_lag_vld>=tol_lag) + f' epoch.{epoch:d}/{num_epochs:d}, lag.{epoch_lag_vld:d}/{tol_lag:d}!'
        print('stop_state: ', stop_state)

        pearson_coef_train = person_coeffcicient(yrstore.detach().numpy(), train_dset[:][1].detach().numpy())
        pearson_coef_valid = person_coeffcicient(yrstore_vld.detach().numpy(), valid_dset[:][1].detach().numpy())

        Err_train = np.array(Err_train)
        Err_valid = np.array(Err_valid)
        Err_train_mav = np.array(Err_train_mav)
        Err_valid_mav = np.array(Err_valid_mav)
        Ave_Grad = np.array(Ave_Grad)

    return {
        'model': mymlp,
        'model_class': type(mymlp),
        'initial_parameters': init_para,
        'valid_slct_parameters': mdlslct_sofar_vld,
        'train_slct_parameters': mdlslct_sofar,
        'valid_err': min_vlderr_sofar,
        'valid_idx': min_sofar_idx_vld,
        'min_train_err': min_err_sofar,
        'min_err_idx': min_sofar_idx,
        'stop_state': stop_state,
        'pearson_train': pearson_coef_train,
        'pearson_valid': pearson_coef_valid,
        'training_evo': {
            'epoch': Out_epochs,
            'train_err': Err_train,
            'train_err_mav': Err_train_mav,
            'valid_err': Err_valid,
            'valid_err_mav': Err_valid_mav,
            'Grad_norm': Ave_Grad,
            'Y_predict': Y_progress
        }
    }

def Plot_MLP_PostTraining(mymlp, R_Mlp, train_dset, valid_dset, title_str='XX: '):
    fig_y, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    fig_y.suptitle(title_str+'y vs target')
    ax1.set_title('training set')
    ax2.set_title('validation set')
    ax1.grid(True, which='both')
    ax2.grid(True, which='both')
    with torch.no_grad():
        mymlp.load_state_dict(R_Mlp['initial_parameters'])
        mymlp.eval()
        ytrn_pre = mymlp(train_dset[:][0])
        yvld_pre = mymlp(valid_dset[:][0])
        ax1.plot(train_dset[:][1].detach(), ytrn_pre.detach(), 'rx', ms=7, mfc='none', label='pre-train')
        ax2.plot(valid_dset[:][1].detach(), yvld_pre.detach(), 'rx', ms=7, mfc='none', label='pre-train')
        ax1.plot(train_dset[:][1].detach(), train_dset[:][1].detach(), 'k-', lw=2, alpha=0.6)
        ax2.plot(valid_dset[:][1].detach(), valid_dset[:][1].detach(), 'k-', lw=2, alpha=0.6)

        mymlp.load_state_dict(R_Mlp['valid_slct_parameters'])
        mymlp.eval()
        ytrn_post = mymlp(train_dset[:][0])
        yvld_post = mymlp(valid_dset[:][0])
        ax1.plot(train_dset[:][1].detach(), ytrn_post.detach(), 'bo', ms=7, mfc='none', label='post-train')
        ax2.plot(valid_dset[:][1].detach(), yvld_post.detach(), 'bo', ms=7, mfc='none', label='post-train')

        ax2.legend()

    R_trn_evo = R_Mlp['training_evo']
    Out_epochs = R_trn_evo['epoch']
    Err_train = R_trn_evo['train_err']
    Err_valid = R_trn_evo['valid_err']
    Err_train_mav = R_trn_evo['train_err_mav']
    Err_valid_mav = R_trn_evo['valid_err_mav']

    fig_err, ax_err = plt.subplots()
    fig_err.suptitle(title_str+'Err training evo')
    ax_err.plot(Out_epochs, Err_train, 'rx', ms=7, mfc='none', alpha=0.6, label='trn err')
    ax_err.plot(Out_epochs, Err_valid, 'go', ms=7, mfc='none', alpha=0.6, label='vld err')
    ax_err.plot(Out_epochs, Err_train_mav, 'b-', lw=1.3, label='trn err mav')
    ax_err.plot(Out_epochs, Err_valid_mav, 'm-', lw=1.3, label='vld err mav')
    ax_err.legend()
    ax_err.set_xscale('log')
    ax_err.set_yscale('log')

    In_Coef_tensor = list(R_Mlp['valid_slct_parameters'].items())[0][1]
    n_hd, n_in = In_Coef_tensor.shape
    In_Coef = np.copy(In_Coef_tensor.detach().numpy())
    Av_Coef = np.mean(In_Coef, axis=0)
    Av_Abs_Coef = np.mean(np.abs(In_Coef), axis=0)
    Max_Coef = np.max(In_Coef, axis=0)
    Min_Coef = np.min(In_Coef, axis=0)
    fig_c, (ax_c1, ax_c2, ax_c3, ax_c4)  = plt.subplots(4,1, figsize=(6,9), sharex=True)
    fig_c.suptitle(title_str+'Input Layer Coeff.')
    clrs = cm.rainbow(np.linspace(0,1, n_hd))
    for i in range(n_hd):
        ax_c1.plot(In_Coef[i,:], 'o', lw=0.6, ms=6, mew=1, mfc='none', color=clrs[i])
    ax_c4.plot(Max_Coef, 'mx--', lw=0.6, ms=6, mew=1, mfc='none')
    ax_c4.plot(Min_Coef, 'c+--', lw=0.6, ms=6, mew=1, mfc='none')
    ax_c2.plot(Av_Coef, 'rs', lw=0.8, ms=8, mew =1.5, mfc='none', label='Ave Coef.')
    ax_c3.plot(Av_Abs_Coef, 'b^', lw=0.8, ms=8, mew =1.5, mfc='none', label='Ave Abs. Coef.')
    ax_c2.legend()
    ax_c3.legend()

    return {
        'fig_y':{
            'fig': fig_y, 
            'axes': [ax1, ax2]
        },
        'fig_err': {
            'fig': fig_err, 
            'axes': [ax_err]
        },
        'fig_coef': {
            'fig': fig_c,
            'axes': [ax_c1, ax_c2, ax_c3, ax_c4]
        }
    }

def main_MLP_study_withGroundTruth():
    pass 

    N=400
    M=5
    ### polynomial data ###
    the_bias = 0.0
    the_coefs = [1,2,0,0,0,0]
    m=M
    features, tarts = pt_study2.Polynomial_Data_Generator(N, M, m, the_coefs, the_bias, noise=0.5)
    # features, tarts = pt_study2.Sine_Data_Generator(N, M, noise=0.5)
    targets = np.array([item[0] for item in tarts]) 
    num_trial, Data_Cluster = pt_study2.RandomSplit_CrossVal_Preparation(N, features, targets, set_num_trials=1.0, set_valid_ratio=0.1)

    Data_Cluster_Tensor = []
    for i, item in enumerate(Data_Cluster):
        data_dict = {}
        for ky1 in item.keys():
            sub_dict = {}
            sub_dict['feature'] = torch.tensor(item[ky1]['feature'])
            sub_dict['target'] = torch.tensor([[val] for val in item[ky1]['target']])
            data_dict[ky1] = sub_dict
        Data_Cluster_Tensor.append(data_dict)

    ### Ridge MAP ###
    # Penalties = np.logspace(-4,np.log10(3),21)
    # print('Ridge MAP usual')
    # R_Ridge = glh_map.Model_Training(num_trial, Data_Cluster, M, regularisation='ridge', Panelty_Values=Penalties)
    # glh_map.Plot_One_Training_Result(R_Ridge)

    ### MLP ###
    ftrs_train = Data_Cluster_Tensor[0]['fit']['feature']
    trgs_train = Data_Cluster_Tensor[0]['fit']['target']
    ftrs_valid = Data_Cluster_Tensor[0]['valid']['feature']
    trgs_valid = Data_Cluster_Tensor[0]['valid']['target']

    train_dset = myDataset_from_tensors(ftrs_train, trgs_train)
    valid_dset = myDataset_from_tensors(ftrs_valid, trgs_valid)

    train_dloader = DataLoader(train_dset, batch_size=int(len(train_dset)/30), shuffle=1)
    valid_dloader = DataLoader(valid_dset, batch_size=1)

    arch = [M,2,1]
    # model_mlp = NeuralNet_MLP(M, 4, 1, arch=arch)
    mymlp = NeuralNet_MLP_Arch(arch=arch)

    ### init ###
    # para_init = collections.OrderedDict([
    #     ('in_lyr.weight', tensor([[-0.4167, -0.2191, -0.3936, -0.2638,  0.0417], [-0.0316, -0.4235,  0.1967, -0.0696, -0.0509]], dtype=torch.float64)),
    #     ('in_lyr.bias', tensor([-0.3154, -0.1973], dtype=torch.float64)),
    #     ('out_lyr.weight', tensor([[-0.3162, -0.2036]], dtype=torch.float64)),
    #     ('out_lyr.bias', tensor([6.1815], dtype=torch.float64))
    # ])
    # mymlp.load_state_dict(para_init)
    # mymlp.eval()

    lr = 1e-3

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(mymlp.parameters(), lr=lr)

    fig_y, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    fig_y.suptitle('y vs target')
    ax1.set_title('training set')
    ax2.set_title('validation set')
    ax1.grid(True, which='both')
    ax2.grid(True, which='both')
    with torch.no_grad():
        ytrn_pre = mymlp(train_dset[:][0])
        yvld_pre = mymlp(valid_dset[:][0])
        ax1.plot(train_dset[:][1].detach(), ytrn_pre.detach(), 'rx', ms=7, mfc='none', label='pre-train')
        ax2.plot(valid_dset[:][1].detach(), yvld_pre.detach(), 'rx', ms=7, mfc='none', label='pre-train')
        ax1.plot(train_dset[:][1].detach(), train_dset[:][1].detach(), 'k-', lw=2, alpha=0.6)
        ax2.plot(valid_dset[:][1].detach(), valid_dset[:][1].detach(), 'k-', lw=2, alpha=0.6)

    ## training ##
    Out_epochs = []
    Y_progress = []
    Err_train = []
    Err_valid = []
    Ave_Grad = []
    Err_train_mav = []
    Err_valid_mav = []
    mav_smpl_size = 300

    num_epochs = 10000
    out_period = 1
    print_step = 200
    num_steps = len(train_dloader)
    tol_lag = int(0.4*num_epochs)
    
    ave_abs_grad = 1e20
    train_err = 1e10
    min_err_sofar = 1e10
    min_vlderr_sofar = 1e10
    epoch_lag = 1
    epoch_lag_vld = 1
    min_sofar_idx = 0
    min_sofar_idx_vld = 0
    mdlslct_sofar = mymlp.dump_state_dict()
    mdlslct_sofar_vld = mymlp.dump_state_dict()
    epoch = 0
    
    # for epoch in range(num_epochs):
    # while epoch<num_epochs and min_err_sofar>err_crt and (min_err_sofar>err_tol or epoch_lag<tol_lag):
    while epoch<num_epochs and epoch_lag_vld<tol_lag:
        for i, (f,t) in enumerate(train_dloader):

            yfwd = mymlp(f)
            err = loss(yfwd, t)
            # yfwd = mymlp(train_dset[:][0])
            # err = loss(yfwd, train_dset[:][1])
            err.backward()
            optimizer.step()
            optimizer.zero_grad()

            # if i%print_step==0:
            #     print('Epoch.{0:d}/{1:d}, Step.{2:d}/{3:d}, Loss.{4:.8f}'.format(epoch, num_epochs, i, num_steps, err.item()))

        Out_epochs.append(epoch)

        with torch.no_grad():
            y_inter = mymlp(train_dset[:][0]).detach()
            err_inter = loss(y_inter, train_dset[:][1])
            train_err = err_inter.item()
            Err_train.append(train_err)
            Y_progress.append(y_inter)

            y_itr_valid = mymlp(valid_dset[:][0])
            err_itr_valid = loss(y_itr_valid, valid_dset[:][1])
            valid_err = err_itr_valid.item()
            Err_valid.append(valid_err)

        if min_err_sofar>train_err:
            min_err_sofar = train_err
            min_sofar_idx = epoch
            mdlslct_sofar = mymlp.dump_state_dict()
        
        if min_vlderr_sofar>valid_err:
            min_vlderr_sofar = valid_err
            min_sofar_idx_vld = epoch
            mdlslct_sofar_vld = mymlp.dump_state_dict()
        
        epoch_lag = epoch - min_sofar_idx
        epoch_lag_vld = epoch - min_sofar_idx_vld

        si = max(0, epoch-mav_smpl_size)
        Err_train_mav.append(np.mean(np.array(Err_train[si:])))
        Err_valid_mav.append(np.mean(np.array(Err_valid[si:])))

        if epoch%print_step==0:
            print(f'Epoch.{epoch:d}/{num_epochs:d}, Loss.{train_err:.8E}, min_err. t.{min_err_sofar:.8E}, v.{min_vlderr_sofar:.8E}, lag.{epoch_lag_vld:d} /{tol_lag:d}')

        epoch = epoch + 1
    else:
        print(f'Epoch: [{epoch:d}/{num_epochs:d}], train_err.{train_err:.4E}, min_err_trn.{min_err_sofar:.4E}, min_err_vld.{min_vlderr_sofar:.4E}, lag. t.{epoch_lag:d} & v.{epoch_lag_vld:d} / {tol_lag:d}')
        print('restore the best selected model')
        mymlp.load_state_dict(mdlslct_sofar)
        mymlp.eval()
        with torch.no_grad():
            yrstore = mymlp(train_dset[:][0])
            lrstore = loss(yrstore, train_dset[:][1])
        print(f'restored train err: {lrstore.item():.8E}')

    Err_train = np.array(Err_train)
    Err_valid = np.array(Err_valid)
    Err_train_mav = np.array(Err_train_mav)
    Err_valid_mav = np.array(Err_valid_mav)
    Ave_Grad = np.array(Ave_Grad)

    ## validation ##
    with torch.no_grad():
        ytrn_post = mymlp(train_dset[:][0])
        yvld_post = mymlp(valid_dset[:][0])
        ax1.plot(train_dset[:][1].detach(), ytrn_post.detach(), 'bo', ms=7, mfc='none', label='post-train')
        ax2.plot(valid_dset[:][1].detach(), yvld_post.detach(), 'bo', ms=7, mfc='none', label='post-train')
        ax2.legend()

        tvld = valid_dset[:][1]
        err_vld = loss(yvld_post, tvld)
        print('Valid loss.{:.8f}'.format(err_vld.item()))

    fig_err, ax_err = plt.subplots()
    fig_err.suptitle('Err training evo')
    ax_err.plot(Out_epochs, Err_train, 'rx', ms=7, alpha=0.6)
    ax_err.plot(Out_epochs, Err_valid, 'g+', ms=7, alpha=0.6)
    ax_err.plot(Out_epochs, Err_train_mav, 'b-', lw=1.3)
    ax_err.plot(Out_epochs, Err_valid_mav, 'm-', lw=1.3)
    # move_ave_window = 20
    # mv_err_idx = []
    # mv_err = []
    # for i in range(len(Err_train)-move_ave_window):
    #     mv_err_idx.append(Out_epochs[int(i+move_ave_window/2)])
    #     mv_err.append(np.mean(Err_train[i:i+move_ave_window]))
    # mv_err_idx = np.array(mv_err_idx)
    # mv_err = np.array(mv_err)
    # ax_err.plot(mv_err_idx, mv_err, 'b-', lw=1.5)
    ax_err.set_xscale('log')
    ax_err.set_yscale('log')

    # fig_grad, ax_grad = plt.subplots()
    # fig_grad.suptitle('parameter Gradiant')
    # ax_grad.plot(Out_epochs, Ave_Grad, 'rx', ms=7)
    # move_ave_window = 20
    # mv_grad_idx = []
    # mv_grad = []
    # for i in range(len(Ave_Grad)-move_ave_window):
    #     mv_grad_idx.append(Out_epochs[int(i+move_ave_window/2)])
    #     mv_grad.append(np.mean(Ave_Grad[i:i+move_ave_window]))
    # mv_grad_idx = np.array(mv_grad_idx)
    # mv_grad = np.array(mv_grad)
    # ax_grad.plot(mv_grad_idx, mv_grad, 'b-', lw=1.5)
    # ax_grad.set_xscale('log')
    # ax_grad.set_yscale('log')

    clrs = cm.rainbow(np.linspace(0,1,len(Out_epochs)))
    num_plots = 50
    plot_steps = int(len(Out_epochs)/num_plots)
    figp, axp = plt.subplots()
    for i in range(len(Out_epochs)):
        if i%plot_steps==0:
            axp.plot(train_dset[:][1].detach(), Y_progress[i], 'o', ms=7, mfc='none', color=clrs[i], alpha=0.6)
        elif i==num_epochs-1:
            axp.plot(train_dset[:][1].detach(), Y_progress[i], 'o', ms=7, mfc='none', color=clrs[i], alpha=0.6)

    # animation_yvsy(Y_progress, Out_epochs, train_dset[:][1].detach(), 20)

def main_MLP_study_withGroundTruth_withTrainingModule():
    print('main_MLP_study_withGroundTruth_withTrainingModule')
    pass 

    N=400
    M=5
    ### polynomial data ###
    the_bias = 0.0
    the_coefs = [1,2,0,0,0,0]
    m=M
    features, tarts = pt_study2.Polynomial_Data_Generator(N, M, m, the_coefs, the_bias, noise=0.5)
    # features, tarts = pt_study2.Sine_Data_Generator(N, M, noise=0.5)
    targets = np.array([item[0] for item in tarts]) 
    num_trial, Data_Cluster = pt_study2.RandomSplit_CrossVal_Preparation(N, features, targets, set_num_trials=1.0, set_valid_ratio=0.1)

    Data_Cluster_Tensor = []
    for i, item in enumerate(Data_Cluster):
        data_dict = {}
        for ky1 in item.keys():
            sub_dict = {}
            sub_dict['feature'] = torch.tensor(item[ky1]['feature'])
            sub_dict['target'] = torch.tensor([[val] for val in item[ky1]['target']])
            data_dict[ky1] = sub_dict
        Data_Cluster_Tensor.append(data_dict)

    ### Ridge MAP ###
    # Penalties = np.logspace(-4,np.log10(3),21)
    # print('Ridge MAP usual')
    # R_Ridge = glh_map.Model_Training(num_trial, Data_Cluster, M, regularisation='ridge', Panelty_Values=Penalties)
    # glh_map.Plot_One_Training_Result(R_Ridge)

    ### MLP ###
    arch = [M,2,1]
    mymlp = NeuralNet_MLP_Arch(arch=arch)
    ### init ###
    # para_init = collections.OrderedDict([
    #     ('in_lyr.weight', tensor([[-0.4167, -0.2191, -0.3936, -0.2638,  0.0417], [-0.0316, -0.4235,  0.1967, -0.0696, -0.0509]], dtype=torch.float64)),
    #     ('in_lyr.bias', tensor([-0.3154, -0.1973], dtype=torch.float64)),
    #     ('out_lyr.weight', tensor([[-0.3162, -0.2036]], dtype=torch.float64)),
    #     ('out_lyr.bias', tensor([6.1815], dtype=torch.float64))
    # ])
    # mymlp.load_state_dict(para_init)
    # mymlp.eval()
    
    lr = 1e-3
    # loss = nn.MSELoss()
    # optimizer = torch.optim.Adam

    ftrs_train = Data_Cluster_Tensor[0]['fit']['feature']
    trgs_train = Data_Cluster_Tensor[0]['fit']['target']
    ftrs_valid = Data_Cluster_Tensor[0]['valid']['feature']
    trgs_valid = Data_Cluster_Tensor[0]['valid']['target']

    train_dset = myDataset_from_tensors(ftrs_train, trgs_train)
    valid_dset = myDataset_from_tensors(ftrs_valid, trgs_valid)

    num_batch = 30
    train_dloader = DataLoader(train_dset, batch_size=int(len(train_dset)/num_batch), shuffle=1)
    valid_dloader = DataLoader(valid_dset, batch_size=1)

    # with torch.no_grad():
    #     ytrn_pr = mymlp(train_dset[:][0])
    #     yvld_pr = mymlp(valid_dset[:][0])

    R_Mlp = Train_MLP_Arch(
        mymlp=mymlp,
        loss=nn.MSELoss(),
        nnOptmzr=torch.optim.Adam,
        lr=lr,
        train_dset=train_dset,
        valid_dset=valid_dset,
        train_dloader=train_dloader,
        valid_dloader=valid_dloader,
        mav_smpl_size=300,
        num_epochs=6000,
        tol_lag_r=0.3
    )

    plots = Plot_MLP_PostTraining(mymlp, R_Mlp, train_dset, valid_dset)

if __name__ =='__main__':
    print('Pytorch_study3.py')
    # sys.exit()

    # main_MLP_study_withGroundTruth()
    # sys.exit()

    # main_MLP_study_withGroundTruth_withTrainingModule()
    # sys.exit()

    ### Global Parameters ###
    data_std_cut = 1e-4 # for neglecting non-varying components in feature data - dr.load_data(..., std_cut=data_std_cut, ...) 

    ### Data Loading & pre-Processing ###
    home_path = '/Users/chenliu/'
    project_path = 'Research_Projects/SVM-SwapMC'
    training_data_path = 'DATA/Training_data'
    training_data_file = 'Cnf2.xy'
    full_data_path = os.path.join(home_path, project_path, training_data_path, training_data_file)

    Ext = glh_map.DataLoad_and_preProcessing(full_data_path, std_cut=data_std_cut)
    N = Ext['total number of data points']
    M = Ext['dimension of input feature vectors']
    features = Ext['input features']
    targets = Ext['output targets']
    std_features = Ext['empirical standard deviation of features']
    mean_features = Ext['empirical mean of features']
    feature_names = Ext['feature names']

    cross_validation_method = 'rand_cross'  #  'nest'   #   'segment' #    

    parameter_crossvalidation = {
        'validation_ratio' : 0.2, # its function depends on the opted cross-validation method. 

        ## for Nest Cross Validation ##
        'num_cell' : int(5), # number of cells in the "nest"

        ## for Segmenting Cross Validation ##
        'num_seg' : int(3), # number of segments apart from validation set

        ## for Random Split Cross Validation ##
        'num_split_ratio': 1.0,  # to generate number of trials : num_split_ratio/validation_ratio

        ## Producibility of randomness ##
        'rand_seed': 357828922 , 
        'seeding_or_not': True,   # true for reproducible cross validation data structure using 'rand_seed'

        ## Control Options ##
        'print_info_or_not': True
    }

    ### Prepare Data for Cross Validation ###
    num_trial, Data_Cluster = glh_map.DataPreparation_for_CrossValidation(cross_validation_method, N, features, targets, parameter_crossvalidation)
    
    Data_Cluster_Tensor = []
    for i, item in enumerate(Data_Cluster):
        data_dict = {}
        for ky1 in item.keys():
            sub_dict = {}
            sub_dict['feature'] = torch.tensor(item[ky1]['feature'])
            sub_dict['target'] = torch.tensor([[val] for val in item[ky1]['target']])
            data_dict[ky1] = sub_dict
        Data_Cluster_Tensor.append(data_dict)

    ### Ridge MAP ###
    Penalties = np.logspace(-3,np.log10(100),21)
    print('Ridge MAP usual')
    R_Ridge = glh_map.Model_Training(num_trial, Data_Cluster, M, regularisation='ridge', Panelty_Values=Penalties)
    glh_map.Plot_One_Training_Result(R_Ridge)
        
    ### MLP ###
    arch = [M,2,10,10,1]
    mymlp = NeuralNet_MLP_Arch(arch=arch)
    lr = 5e-3
    loss_func = nn.MSELoss()
    optmzr = torch.optim.Adam

    num_batch = 10
    R_mlp_trials = []
    for i in range(num_trial):
        ftrs_train = Data_Cluster_Tensor[i]['fit']['feature']
        trgs_train = Data_Cluster_Tensor[i]['fit']['target']
        ftrs_valid = Data_Cluster_Tensor[i]['valid']['feature']
        trgs_valid = Data_Cluster_Tensor[i]['valid']['target']

        train_dset = myDataset_from_tensors(ftrs_train, trgs_train)
        valid_dset = myDataset_from_tensors(ftrs_valid, trgs_valid)

        train_dloader = DataLoader(train_dset, batch_size=int(len(train_dset)/num_batch), shuffle=1)
        valid_dloader = DataLoader(valid_dset, batch_size=1)

        mymlp.rand_refresh()
        R_Mlp = Train_MLP_Arch(
            mymlp=mymlp,
            loss=loss_func,
            nnOptmzr=optmzr,
            lr=lr,
            train_dset=train_dset,
            valid_dset=valid_dset,
            train_dloader=train_dloader,
            valid_dloader=valid_dloader,
            tol_lag_r=0.3
        )

        Plot_MLP_PostTraining(mymlp, R_Mlp, train_dset, valid_dset, title_str=f'trial.{i:d}')

        R_mlp_trials.append(R_Mlp)
    
    for i, Ritem in enumerate(R_mlp_trials):
        print(f"Trial.{i:d}/{num_trial:d}, Valid_Err.{Ritem['valid_err']:.4E}, Train_Err.{Ritem['min_train_err']:.4E}, Pearson t/v: {Ritem['pearson_train']:.4E}/{Ritem['pearson_valid']:.4E}, Stop: {Ritem['stop_state']} .")
    
    