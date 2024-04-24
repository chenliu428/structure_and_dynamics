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


import data_read as dr 
import data_prep_crossvalid as dxv
import plot_tools
import GaussianLH_Panelty_RidgeLasso_MAP as glh_map
import DeepLearning_Functionalities as dlf
from DeepLearning_MLP import pearson_coeffcicient as pearson_cf

imp.reload(dr)
imp.reload(dxv)
imp.reload(plot_tools)
imp.reload(glh_map)
imp.reload(dlf)


def file_name(cnf_name, mthd_name):
    return cnf_name+'_'+mthd_name+'_Dict'

def extract_weight_bias_MAP(model_dict):
    rdict = model_dict['result']
    idx = np.where(rdict['Ave_Risk(panelty)'] == np.min(rdict['Ave_Risk(panelty)']))[0][0]
    ky_pnlt = rdict['Panelty'][idx]
    return {
        'ws': rdict['Ave_Ws(panelty)'][ky_pnlt],
        'w0': rdict['Ave_W0(panelty)'][idx]
    }

def extract_weight_bias_Bayes(model_dict):
    return {
        'ws': model_dict['result_all_data']['Ws'],
        'w0': model_dict['result_all_data']['W0']
    }

def extract_weight_bias_MLP(model_dict):
    return model_dict['model'], model_dict['valid_slct_parameters']

def pearson_linear_model(lm_para, num_trial, Data_Cluster):
    pearsons = []
    for i, dset in enumerate(Data_Cluster):
        ftrs = dset['valid']['feature']
        trgs = dset['valid']['target']
        y_prdct = np.dot(ftrs, lm_para['ws']) + lm_para['w0']
        pearsons.append(pearson_cf(y_prdct, trgs))
    pearsons = np.array(pearsons)
    return np.mean(pearsons), np.std(pearsons)

def pearson_mlp(mlp_model:dlf.NeuralNet_MLP_Arch, state_dict_list, Data_Cluster_Tensor):
    num_models = len(state_dict_list)
    pearsons = []
    for i, dset in enumerate(Data_Cluster_Tensor):
        ftrs_valid = dset['valid']['feature']
        trgs_valid = dset['valid']['target']
        valid_dset = dlf.myDataset_from_tensors(ftrs_valid, trgs_valid)

        pearson_c = 0.0
        for j, state_dict in enumerate(state_dict_list):
            mlp_model.load_state_dict(state_dict)
            mlp_model.eval()
            with torch.no_grad():
                y_vld = mlp_model(valid_dset[:][0])
            pearson_c = pearson_c + pearson_cf(y_vld.detach().numpy(), valid_dset[:][1].detach().numpy())
        pearson_c = pearson_c/num_models
        pearsons.append(pearson_c)

    pearsons = np.array(pearsons)
    return np.mean(pearsons), np.std(pearsons)

if __name__=='__main__':

    print('Performance comparison')

    Cnf_name='Cnf2.xy'

    ### Global Parameters ###
    data_std_cut = 1e-4 # for neglecting non-varying components in feature data - dr.load_data(..., std_cut=data_std_cut, ...) 

    ### Data Loading & pre-Processing ###
    home_path = ''
    project_path = './'
    training_data_path = 'DATA/Training_data'
    # Cnf_name='Cnf2.xy'
    training_data_file = Cnf_name 
    full_data_path = os.path.join(home_path, project_path, training_data_path, training_data_file)

    Ext = dr.DataLoad_and_preProcessing(full_data_path, std_cut=data_std_cut)
    N = Ext['total number of data points']
    M = Ext['dimension of input feature vectors']
    features = Ext['input features']
    targets = Ext['output targets']
    std_features = Ext['empirical standard deviation of features']
    mean_features = Ext['empirical mean of features']
    feature_names = Ext['feature names']

    cross_validation_method = 'rand_cross'  #  'nest'   #   'segment' #    

    parameter_crossvalidation = {
        'validation_ratio' : 0.6, # its function depends on the opted cross-validation method. 

        ## for Nest Cross Validation ##
        'num_cell' : int(5), # number of cells in the "nest"

        ## for Segmenting Cross Validation ##
        'num_seg' : int(3), # number of segments apart from validation set

        ## for Random Split Cross Validation ##
        'num_split_ratio': 6.0,  # to generate number of trials : num_split_ratio/validation_ratio

        ## Producibility of randomness ##
        'rand_seed': 357828922 , 
        'seeding_or_not': True,   # true for reproducible cross validation data structure using 'rand_seed'

        ## Control Options ##
        'print_info_or_not': True
    }

    ### Prepare Data for Cross Validation ###
    num_trial, Data_Cluster = dxv.DataPreparation_for_CrossValidation(cross_validation_method, N, features, targets, parameter_crossvalidation)
    
    Data_Cluster_Tensor = []
    for i, item in enumerate(Data_Cluster):
        data_dict = {}
        for ky1 in item.keys():
            sub_dict = {}
            sub_dict['feature'] = torch.tensor(item[ky1]['feature'])
            sub_dict['target'] = torch.tensor([[val] for val in item[ky1]['target']])
            data_dict[ky1] = sub_dict
        Data_Cluster_Tensor.append(data_dict)

    ### Load models ###
    Deep_Archs=[
        [M,2,1],
        [M,5,5,1],
        [M,10,10,1],
        [M,2,10,10,1]
    ]

    mthd_keys = ['MAP_ridge', 'Bayes_ridge', 'MAP_lasso', 'MAP_debias']
    for item in Deep_Archs:
        ky = 'Deep_Arch_In.'
        for i, n in enumerate(item):
            ky = ky + '-'*(i!=0) + f'{n:d}'
        mthd_keys.append(ky)
    
    class_linear_map = [item for item in mthd_keys if 'MAP' in item]
    class_linear_bayes = [item for item in mthd_keys if 'Bayes' in item]
    class_mlp = [item for item in mthd_keys if 'Deep_Arch' in item]

    model_dicts = {}
    for i, mthd_ky in enumerate(mthd_keys):
        model_dicts[mthd_ky] = dr.read_a_dictionary_file('./DATA/Results_data/'+file_name(Cnf_name, mthd_ky))

    ### compute pearson coefficients ###
    Pearson_Coeff={'mean':{},'std':{}}
    for i, mthd_ky in enumerate(mthd_keys):
        
        mean_pearson = 0.0
        std_pearson = 0.0
        
        if mthd_ky in class_linear_map:
            lm_para = extract_weight_bias_MAP(model_dicts[mthd_ky])
            if 'debias' in mthd_ky:
                full_fn = model_dicts['MAP_ridge']['feature names']
                dbis_fn = model_dicts[mthd_ky]['feature names']
                new_ws = np.zeros(M)
                for k, fn in enumerate(full_fn):
                    if fn in dbis_fn:
                        db_idx = dbis_fn.index(fn)
                        new_ws[k] = lm_para['ws'][db_idx]
                lm_para['ws'] = new_ws
            mean_pearson, std_pearson = pearson_linear_model(lm_para, num_trial, Data_Cluster)
        
        elif mthd_ky in class_linear_bayes:
            lm_para = extract_weight_bias_Bayes(model_dicts[mthd_ky])
            mean_pearson, std_pearson = pearson_linear_model(lm_para, num_trial, Data_Cluster)

        elif mthd_ky in class_mlp:
            mlp_model, state_dict_list = extract_weight_bias_MLP(model_dicts[mthd_ky])
            mean_pearson, std_pearson = pearson_mlp(mlp_model, state_dict_list, Data_Cluster_Tensor)

        else:
            print('error method key')
            sys.exit()
        
        Pearson_Coeff['mean'][mthd_ky] = mean_pearson
        Pearson_Coeff['std'][mthd_ky] = std_pearson

    xtick_labels = list(Pearson_Coeff['mean'].keys())
    pearson_vlu = list(Pearson_Coeff['mean'].values())
    pearson_err = list(Pearson_Coeff['std'].values())
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes([0.15, 0.25, 0.8, 0.7])
    fig.suptitle(f'Pearson Coefficients')
    ax.errorbar(xtick_labels, pearson_vlu, yerr=pearson_err, fmt='d', mfc='none', ms=15, mew=1, elinewidth=1, capsize=12, color='r')
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')
