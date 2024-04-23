import math as ma
import numpy as np
from numpy import copy
from numpy import *

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from scipy import *
from scipy import fftpack as ftp
from scipy.optimize import curve_fit, leastsq
from scipy import interpolate as interplt
from scipy.interpolate import interp1d

import sys
from datetime import datetime
import os.path
import imp
import sklearn.cluster as skcltr
import sklearn.linear_model as skl_linear

# for data loading
import data_read as dr 

# for cross-validation data preparation 
import data_prep_crossvalid as dxv

# handmade linear regressors
import simpleLinearReg as slr

# plot tools
import plot_tools

imp.reload(dr)
imp.reload(dxv)
imp.reload(slr)
imp.reload(plot_tools)

# SqErr_func = lambda x,y : np.mean((x-y)*(x-y))
# func = lambda x, w, w0 , r : np.dot(w, x) + w0 + r
Linear_model = lambda x, w, w0: np.dot(w, x) + w0

def Model_Training(num_trial, Data_Cluster, M, regularisation, Panelty_Values):

    Risk_for_Panelty = []
    Risk_Spr_Panelty = []
    Nll_for_Panelty =[]
    Nll_Spr_Panelty = []
    Ws_for_Panelty = {}
    W0_for_Panelty = []
    Ws_lists = {}
    W0_lists = {}
    Risk_list = []
    Nll_list =[]
    for idx_pnlt in range(len(Panelty_Values)):
        pnlt = Panelty_Values[idx_pnlt]
        Ws_of_cells = []
        W0_of_cells = []
        Pr_of_cells = []
        Risk_of_cells = []
        Nll_of_cells = []
        for idx_cell in range(num_trial):
            print('idx_pnlt {0:d} / {1:d}, idx_cell {2:d} / {3:d}'.format(idx_pnlt, len(Panelty_Values), idx_cell, num_trial))
            targets_to_fit = Data_Cluster[idx_cell]['fit']['target']
            features_to_fit = Data_Cluster[idx_cell]['fit']['feature']
            targets_valid = Data_Cluster[idx_cell]['valid']['target']
            features_valid = Data_Cluster[idx_cell]['valid']['feature']
            N_fit = len(targets_to_fit)

            if regularisation == 'ridge':
                Lr = slr.Linear_Regression_Ridge_HomeMade(targets=targets_to_fit, features=features_to_fit, panelty_coeff_ratio=pnlt, stop_crt=1e-5, max_iter_in_N=100, dt=0.0001, init_coeffs=np.ones(M), init_offset=0, method='matrix inverse', plot_info=1)
                ws_mti = Lr['weights']
                w0_mti = Lr['offset']
                pr_mti = np.array([ Linear_model(features_valid[i,:], ws_mti, w0_mti) for i in range(len(targets_valid))])
                risk = np.mean((pr_mti-targets_valid)**2)
                nll = 0.5*np.log(risk)+0.5*(1.0+np.log(2*np.pi))
            elif regularisation == 'lasso':
                lasso_alpha = pnlt/N_fit
                # print('lasso alpha: ', lasso_alpha)
                clf = skl_linear.Lasso(alpha=lasso_alpha, fit_intercept=True, max_iter=1000000, tol=1e-5, warm_start=False) 
                clf.fit(features_to_fit, targets_to_fit)
                ws_mti = clf.coef_
                w0_mti = clf.intercept_
                pr_mti = clf.predict(features_valid)
                pr_mti1 = np.array([ Linear_model(features_valid[i,:], ws_mti, w0_mti) for i in range(len(targets_valid))])
                risk = np.mean((pr_mti-targets_valid)**2)
                nll = 0.5*np.log(risk)+0.5*(1.0+np.log(2*np.pi))
                # print('pr-pr1:', np.mean(np.abs(pr_mti-pr_mti1)))
            else:
                print('wrong option for regularisation')
                sys.exit()

            Ws_of_cells.append(ws_mti)
            W0_of_cells.append(w0_mti)
            Pr_of_cells.append(pr_mti)
            Risk_of_cells.append(risk)
            Nll_of_cells.append(nll)

            # if idx_cell==0:
            #     plt.figure()
            #     plt.title('Pr vs Target - Panelty = {0:.2f}, cell = {1:d}'.format(pnlt, idx_cell))
            #     plt.plot(targets_valid, pr_mti, 'o', ms=8, mew=1, mfc='none')
            #     plt.plot(targets_valid, targets_valid, 'k--', lw=1)

        Risk_of_cells = np.array(Risk_of_cells)
        Nll_of_cells = np.array(Nll_of_cells)
        W0_of_cells = np.array(W0_of_cells)

        Ave_Risk = np.mean(Risk_of_cells)
        Std_Risk = np.std(Risk_of_cells)
        Ave_Nll = np.mean(Nll_of_cells)
        Std_Nll = np.std(Nll_of_cells)
        Ave_W0 = np.mean(W0_of_cells)
        Ave_Ws = np.zeros(M)
        for k in range(num_trial):
            Ave_Ws = Ave_Ws + Ws_of_cells[k]/float(num_trial)
        
        Risk_for_Panelty.append(Ave_Risk)
        # Risk_for_Panelty.append(np.mean( np.array([(Linear_model(features[j,:], Ave_Ws, Ave_W0) - targets[j])**2 for j in range(N)] )))
        Risk_Spr_Panelty.append(Std_Risk)
        Nll_for_Panelty.append(Ave_Nll)
        Nll_Spr_Panelty.append(Std_Nll)
        Ws_for_Panelty[pnlt] = Ave_Ws
        W0_for_Panelty.append(Ave_W0)
        Ws_lists[pnlt] = Ws_of_cells
        W0_lists[pnlt] = W0_of_cells
        Risk_list.append(Risk_of_cells) 
        Nll_list.append(Nll_of_cells)

    W0_for_Panelty = np.array(W0_for_Panelty)

    Risk_for_Panelty = np.array(Risk_for_Panelty)
    Risk_Spr_Panelty = np.array(Risk_Spr_Panelty)
    Risk_list = np.array(Risk_list)
    Nll_for_Panelty = np.array(Nll_for_Panelty)
    Nll_Spr_Panelty = np.array(Nll_Spr_Panelty)
    Nll_list = np.array(Nll_list)

    Ws_Fluc_Panelty = {'std':[], 'max-min':[]}
    W0_Fluc_Panelty = {'std':[], 'max-min':[]}
    for i in range(len(Panelty_Values)):
        pnlt = Panelty_Values[i]
        Ws_ = np.array(Ws_lists[pnlt])
        W0_ = W0_lists[pnlt]
        std_ws = np.array([np.std(Ws_[:,k]) for k in range(M)])
        spr_ws = np.array([np.max(Ws_[:,k])-np.min(Ws_[:,k]) for k in range(M)])
        std_w0 = np.std(W0_)
        spr_w0 = np.max(W0_)-np.min(W0_) 
        Ws_Fluc_Panelty['std'].append(std_ws)
        Ws_Fluc_Panelty['max-min'].append(spr_ws)
        W0_Fluc_Panelty['std'].append(std_w0)
        W0_Fluc_Panelty['max-min'].append(spr_w0)

    for key in Ws_Fluc_Panelty.keys():
        Ws_Fluc_Panelty[key] = np.array(Ws_Fluc_Panelty[key])
        W0_Fluc_Panelty[key] = np.array(W0_Fluc_Panelty[key])

    return {
        'Panelty': Panelty_Values,
        'Ave_W0(panelty)': W0_for_Panelty, 
        'Std_W0(panelty)': W0_Fluc_Panelty,
        'Ave_Ws(panelty)': Ws_for_Panelty,
        'Std_Ws(panelty)':Ws_Fluc_Panelty,
        'Ave_Risk(panelty)': Risk_for_Panelty,
        'Std_Risk(panelty)': Risk_Spr_Panelty,
        'Ave_Nll(panelty)': Nll_for_Panelty,
        'Std_Nll(panelty)': Nll_Spr_Panelty,
        'Risk_list': Risk_list,
        'Nll_list': Nll_list
    }

def Debiasing_Training(num_trial, Data_Cluster, Panelty_Values, R_Lasso):
    
    pnlt_lasso = R_Lasso['Panelty']
    optimal_idx_lasso = np.where(R_Lasso['Ave_Risk(panelty)']==np.min(R_Lasso['Ave_Risk(panelty)']))[0][0]
    Ws_Ave_Lasso = R_Lasso['Ave_Ws(panelty)'][pnlt_lasso[optimal_idx_lasso]]
    idxs_kept = []
    for i in range(len(Ws_Ave_Lasso)):
        if np.abs(Ws_Ave_Lasso[i])>0: idxs_kept.append(i)
    
    new_M = len(idxs_kept)
    # Reshape Data_Cluster
    New_Data_Cluster = []
    for i in range(len(Data_Cluster)):
        N_fit = len(Data_Cluster[i]['fit']['target'])
        N_val = len(Data_Cluster[i]['valid']['target'])
        new_item = {'fit':{}, 'valid': {}}
        fit_features = np.zeros([N_fit, new_M])
        val_features = np.zeros([N_val, new_M])
        for j in range(len(idxs_kept)):
            idx = idxs_kept[j]
            fit_features[:,j] = Data_Cluster[i]['fit']['feature'][:,idx]
            val_features[:,j] = Data_Cluster[i]['valid']['feature'][:,idx]
        new_item['fit']['feature'] = copy(fit_features)
        new_item['valid']['feature'] = copy(val_features)
        new_item['fit']['target'] = copy(Data_Cluster[i]['fit']['target'])
        new_item['valid']['target'] = copy(Data_Cluster[i]['valid']['target'])
        New_Data_Cluster.append(new_item)
    
    return Model_Training(num_trial=num_trial, Data_Cluster=New_Data_Cluster, M=new_M, regularisation='ridge', Panelty_Values=Panelty_Values), np.array(idxs_kept)

def main_MAP_RidgeLassoDebias(cnf_name:str):
    ### Global Parameters ###
    data_std_cut = 1e-4 # for neglecting non-varying components in feature data - dr.load_data(..., std_cut=data_std_cut, ...) 
    
    # Pnlt = {
    #     'ridge': np.logspace(-2,8,31),
    #     'lasso': np.logspace(1,8,31),
    #     'debias': np.logspace(-6,8,43),
    # }

    Pnlt = {
        'ridge': np.append(np.array([0.005, 0.01, 0.03, 0.06,0.09,0.1, 0.2,0.5,1,2, 3,5,10, 15, 19]), np.linspace(20,200,11)), 
        'lasso': np.linspace(10, 100, 21)
    }
    Pnlt.update({'debias': np.append(np.logspace(-6,-3,7), Pnlt['ridge'])})

    cross_validation_method = 'rand_cross'  #  'nest'   #   'segment' #    

    parameter_crossvalidation = {
        'validation_ratio' : 0.1, # its function depends on the opted cross-validation method. 

        ## for Nest Cross Validation ##
        'num_cell' : int(5), # number of cells in the "nest"

        ## for Segmenting Cross Validation ##
        'num_seg' : int(3), # number of segments apart from validation set

        ## for Random Split Cross Validation ##
        'num_split_ratio': 1.5,  # to generate number of trials : num_split_ratio/validation_ratio

        ## Producibility of randomness ##
        'rand_seed': 357828922 , 
        'seeding_or_not': True,   # true for reproducible cross validation data structure using 'rand_seed'

        ## Control Options ##
        'print_info_or_not': True
    }

    ### Data Loading & pre-Processing ###
    home_path = '/Users/chenliu/'
    project_path = 'Research_Projects/SVM-SwapMC'
    training_data_path = 'DATA/Training_data'
    training_data_file = cnf_name # 'Cnf1.xy'
    full_data_path = os.path.join(home_path, project_path, training_data_path, training_data_file)

    Ext = dr.DataLoad_and_preProcessing(full_data_path, std_cut=data_std_cut)
    N = Ext['total number of data points']
    M = Ext['dimension of input feature vectors']
    features = Ext['input features']
    targets = Ext['output targets']
    std_features = Ext['empirical standard deviation of features']
    mean_features = Ext['empirical mean of features']
    feature_names = Ext['feature names']

    ### Prepare Data for Cross Validation ###
    num_trial, Data_Cluster = dxv.DataPreparation_for_CrossValidation(cross_validation_method, N, features, targets, parameter_crossvalidation)

    ### Training ###
    print('Training Ridge')
    R_Ridge = Model_Training(num_trial, Data_Cluster, M, regularisation='ridge', Panelty_Values=Pnlt['ridge'])

    print('Training Lasso')
    R_Lasso = Model_Training(num_trial, Data_Cluster, M, regularisation='lasso', Panelty_Values=Pnlt['lasso'])

    print('Training Debias')
    R_Debias, Indexs_Kept = Debiasing_Training(num_trial, Data_Cluster, Pnlt['debias'], R_Lasso)

    return  {'ridge': R_Ridge, 'lasso': R_Lasso, 'debias': R_Debias}, Indexs_Kept, feature_names

def main_MAP_RidgeLassoDebias_SaveToFile(Cnf_name:str='Cnf2.xy'):
    pass

    # Cnf_name = 'Cnf2.xy'
    print('Training on ', Cnf_name)
    R_main, index_kept, feature_names = main_MAP_RidgeLassoDebias(Cnf_name)
    feature_names_kept = [feature_names[idx] for idx in index_kept]
    
    R_Ridge = R_main['ridge']
    R_Lasso = R_main['lasso']
    R_Dbias = R_main['debias']

    ## save to file ##
    print('Save dicts to file: ')
    for key in R_main.keys():
        print(' - ', key)
        Rdict = R_main[key]
        if key!='debias':
            R_save = {'feature names': feature_names, 'result': Rdict}
        else:
            R_save = {'feature names': feature_names_kept, 'result': Rdict}
        dr.save_to_file_a_dictrionary('./DATA/Results_data/'+Cnf_name+'_MAP_'+key+'_Dict', R_save)

    ## plot results ##
    print('Plot')
    Fig_Objs = plot_tools.Plot_Several_Training_Results(Results_Dict=R_main, Features_Idexs_In={'debias':index_kept}, Data_name=Cnf_name)

def main_ReadResults_and_Plot(Cnf_name:str, methods:list=['ridge', 'lasso', 'debias'], path:str='./DATA/Results_data'):
    Rdct_all = {}
    Ftrs_all = {}
    for item in methods:
        print('item:', item)
        file_path = os.path.join(path, Cnf_name+'_MAP_'+item+'_Dict')
        Rload = dr.read_a_dictionary_file(file_path)
        Rdct_all[item] = Rload['result']
        Ftrs_all[item] = Rload['feature names']
    
    Ft_Indx_all = {}

    num_idx = -100
    full_ftrs = []
    for key in Ftrs_all.keys():
        if len(Ftrs_all[key])>=num_idx:
            num_idx = len(Ftrs_all[key])
            full_ftrs = Ftrs_all[key]
    for key in Ftrs_all.keys():
        indexs = np.array([ full_ftrs.index(item) for item  in Ftrs_all[key] ])
        Ft_Indx_all[key] = indexs

    return plot_tools.Plot_Several_Training_Results(Rdct_all, Ft_Indx_all, Data_name=Cnf_name)    


if __name__ == "__main__":
    
    print('GaussianLH_Panelty_RidgeLasso_MAP.py')
    #sys.exit()

    # main_MAP_RidgeLassoDebias_SaveToFile('Cnf2.xy')
    # sys.exit()

    main_ReadResults_and_Plot('Cnf2.xy')
    sys.exit()

