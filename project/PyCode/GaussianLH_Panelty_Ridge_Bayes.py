import math as ma
import numpy as np
from numpy import copy
from numpy import where
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

imp.reload(dr)
imp.reload(dxv)
imp.reload(slr)

def Bayes_LR_Ridge_UnNormalised_Training_SearchInFeededPenalties(penalties, targets, features):
    El_Q_Un = slr.Bayes_LR_Ridge_UnNormalised_ElemtaryQuantities(targets, features)
    
    Metas_Un = []
    for item in penalties:
        Metas_Un.append(slr.K_related_UnNormalised(El_Q_Un, item))
    
    Meta_R_Un = {key:[] for key in Metas_Un[-1].keys()}
    
    for D in Metas_Un:
        for key in Meta_R_Un.keys():
            Meta_R_Un[key].append(D[key])
    
    for key in Meta_R_Un.keys():
        if type(Meta_R_Un[key][0])!= np.ndarray:
            Meta_R_Un[key] = np.array(Meta_R_Un[key])

    min_idxs = where(Meta_R_Un['nll']==np.min(Meta_R_Un['nll']))[0]
    idx_min_Un = min_idxs[0]

    return {'opt_pnlt': penalties[idx_min_Un], 'W0': Meta_R_Un['Ave_W0'][idx_min_Un], 'Ws': Meta_R_Un['Ave_Ws'][idx_min_Un], 'CVar_Ws':Meta_R_Un['CVar_Ws'][idx_min_Un], 'Penalties': penalties, 'nlls': Meta_R_Un['nll'], 'betas': Meta_R_Un['beta'], 'Num Eff. Ft.': Meta_R_Un['Num Eff. Ft.'], 'All_Ws': Meta_R_Un['Ave_Ws'], 'All_W0': Meta_R_Un['Ave_W0'], 'all_opt_idx':min_idxs, 'the_opt_idx':idx_min_Un}

def Bayes_LR_Lasso_UnNormalised_Training_SearchFeededHypes(penalties, betas, targets, features, mc_size:int=1000):
    El_Q_Un = slr.Bayes_LR_Ridge_UnNormalised_ElemtaryQuantities(targets, features)

    Hype_Tuples = []
    Hype_Coords = {}
    for i in range(len(penalties)):
        for j in range(len(betas)):
            tk = (penalties[i], betas[j])
            Hype_Tuples.append(tk)
            Hype_Coords[tk] = (i,j)
    
    hatWs = slr.generate_hatW_samples(El_Q_Un['M'], mc_size)

    Metas = []
    for i in range(len(Hype_Tuples)):
        p = Hype_Tuples[i][0]
        b = Hype_Tuples[i][1]
        Metas.append(slr.Bayes_LR_Lasso_NLL(El_Q_Un, p, b, hatWs))
    
    Meta_R_Lasso = {key:[] for key in Metas[0].keys()}
    for D in Metas:
        for key in Meta_R_Lasso.keys():
            Meta_R_Lasso[key].append(D[key])

    R_Matrix = {}
    for key in Metas[0].keys():
        if type(Metas[0][key]) == float:
            R_Matrix[key] = np.zeros([len(penalties), len(betas)])
            for i in range(len(Meta_R_Lasso[key])):
                h_tuple = Hype_Tuples[i]
                h_coord = Hype_Coords[h_tuple]
                R_Matrix[key][h_coord[0],h_coord[1]] = Meta_R_Lasso[key][i]

    idx_min = Meta_R_Lasso['nll'].index(np.min(np.array(Meta_R_Lasso['nll'])))
    rt = {}
    for key in Meta_R_Lasso.keys():
        rt[key] = Meta_R_Lasso[key][idx_min]
    rt.update({'Hypes': Hype_Tuples, 'HCoords': Hype_Coords, 'R_Matrix': R_Matrix, 'Metas': Meta_R_Lasso})
    return rt

def Bayes_LR_Lasso_UnNormalised_Training_SearchFeededHypes_2(penalties, betas, targets, features, mc_size:int=1000):
    Elqs = slr.Bayes_LR_Lasso_Un_EQ(targets, features)

    Hype_Tuples = []
    Hype_Coords = {}
    for i in range(len(penalties)):
        for j in range(len(betas)):
            tk = (penalties[i], betas[j])
            Hype_Tuples.append(tk)
            Hype_Coords[tk] = (i,j)
    
    Metas = []
    for i in range(len(Hype_Tuples)):
        p = Hype_Tuples[i][0]
        b = Hype_Tuples[i][1]
        what = slr.generate_W_samples(Elqs['mean_ws'], Elqs['Mtrx_inv'], b, mc_size)
        Metas.append(slr.Bayes_LR_Lasso_NLL_MG(Elqs, p, b, what))
    
    Meta_R_Lasso = {key:[] for key in Metas[0].keys()}
    for D in Metas:
        for key in Meta_R_Lasso.keys():
            Meta_R_Lasso[key].append(D[key])

    R_Matrix = {}
    for key in Metas[0].keys():
        if type(Metas[0][key]) == float:
            R_Matrix[key] = np.zeros([len(penalties), len(betas)])
            for i in range(len(Meta_R_Lasso[key])):
                h_tuple = Hype_Tuples[i]
                h_coord = Hype_Coords[h_tuple]
                R_Matrix[key][h_coord[0],h_coord[1]] = Meta_R_Lasso[key][i]

    nll_mtrx = np.zeros([len(penalties), len(betas)])
    for i in range(len(Meta_R_Lasso['nll'])):
        h_tuple = Hype_Tuples[i]
        h_coord = Hype_Coords[h_tuple]
        nll_mtrx[h_coord[0],h_coord[1]] = Meta_R_Lasso['nll'][i]

    idx_min = Meta_R_Lasso['nll'].index(np.min(np.array(Meta_R_Lasso['nll'])))
    rt = {}
    for key in Meta_R_Lasso.keys():
        rt[key] = Meta_R_Lasso[key][idx_min]
    rt.update({'Hypes': Hype_Tuples, 'HCoords': Hype_Coords, 'R_Matrix': R_Matrix, 'Metas': Meta_R_Lasso, 'nll_mtrx':nll_mtrx, 'opt_Hypes': Hype_Tuples[idx_min]})
    return rt

def Feature_Selection_Analysis():
    dataset_names = [ 'Cnf2.xy', 'Cnf3.xy'] # 'Cnf1.xy',
    methods = {
        'MAP': ['ridge', 'lasso', 'debias'], 
        'Bayes': ['ridge']
    }
    approaches = [item  for item in methods.keys()]

    Clrs={
        'MAP': {'debias':'g', 'ridge':'b', 'lasso': 'r'},
        'Bayes': {'ridge': 'm'}
    }
    Syms={
        'MAP': {'debias':'s', 'ridge':'o', 'lasso': 'D'},
        'Bayes': {'ridge': '^'}
    }

    ## Load the entire file saving linear regression results ##
    path = './DATA/Results_data'
    Files_names = {}
    R_LinReg = {}
    for dname in dataset_names:
        Files_names[dname] = {}
        R_LinReg[dname] = {}
        for app in approaches:
            Files_names[dname][app]={}
            R_LinReg[dname][app]={}
            for reg in methods[app]:
                file_name = dname+'_'+app+'_'+reg+'_Dict'
                R_dict = dr.read_a_dictionary_file(os.path.join(path, file_name))
                Files_names[dname][app][reg] = file_name
                R_LinReg[dname][app][reg] = R_dict

    Ftrs_Names={}
    for i in range(len(dataset_names)):
        dname_key = dataset_names[i]
        R_lr_MAP = R_LinReg[dname_key]['MAP']
        R_lr_BYA = R_LinReg[dname_key]['Bayes']
        full_features = R_lr_MAP['ridge']['feature names']

        ## Bayes data preparation ##
        R_BYA_ridge = R_lr_BYA['ridge']['result_all_data']
        Bya_Ws = R_BYA_ridge['Ws']
        eff_num_coef = int(R_BYA_ridge['Num Eff. Ft.'][R_BYA_ridge['the_opt_idx']]+1)
        sort_ws = np.sort(np.abs(Bya_Ws))
        eff_idx = where(np.abs(Bya_Ws)>=sort_ws[-eff_num_coef])[0]
        eff_cof = np.array([ Bya_Ws[item] for item in eff_idx ])

        Ftrs_Names[dname_key] = {
            'MAP-debias': R_lr_MAP['debias']['feature names'], 
            'Bay-ridge': [full_features[idx] for idx in eff_idx]
        }

    Common_Ftrs_aX_System = {key: set(full_features) for key in Ftrs_Names[dataset_names[0]].keys()}
    Common_Ftrs_aX_Approach = {key: set(full_features) for key in dataset_names}
    for dname in dataset_names:
        for key in Common_Ftrs_aX_System.keys():
            Common_Ftrs_aX_System[key] = Common_Ftrs_aX_System[key].intersection(set(Ftrs_Names[dname][key]))
            Common_Ftrs_aX_Approach[dname] = Common_Ftrs_aX_Approach[dname].intersection(set(Ftrs_Names[dname][key]))
    
    Prcnt_aX_Sys = {}
    Prcnt_aX_App = {}
    for key_app in Common_Ftrs_aX_System.keys():
        Prcnt_aX_Sys[key_app]={}
        Prcnt_aX_App[key_app]={}
        for key_dname in dataset_names:
            Prcnt_aX_Sys[key_app][key_dname] = float(len(Common_Ftrs_aX_System[key_app]))/float(len(Ftrs_Names[key_dname][key_app]))
            Prcnt_aX_App[key_app][key_dname] = float(len(Common_Ftrs_aX_Approach[key_dname]))/float(len(Ftrs_Names[key_dname][key_app]))
    
    Prcnt_aX_Sys_Mtrx = np.zeros([len(Common_Ftrs_aX_System), len(Common_Ftrs_aX_Approach)])
    Prcnt_aX_App_Mtrx = np.zeros([len(Common_Ftrs_aX_System), len(Common_Ftrs_aX_Approach)])
    i=0
    for key_app in Common_Ftrs_aX_System.keys():
        j=0
        for key_dname in Common_Ftrs_aX_Approach.keys():
            Prcnt_aX_Sys_Mtrx[i,j] = Prcnt_aX_Sys[key_app][key_dname]
            Prcnt_aX_App_Mtrx[i,j] = Prcnt_aX_App[key_app][key_dname]
            j=j+1
        i=i+1
    
    #print ax System#
    space = '             '
    sys_list_str = ''
    for ite in dataset_names: sys_list_str = sys_list_str+ite+', '
    sys_list_str=sys_list_str[:-2]
    print('Ftr. Overlap Pct. ax Systems: '+sys_list_str)
    line1 = space
    for key_dname in Common_Ftrs_aX_Approach.keys():
        line1 = line1 + '|' 
        spc = ''
        for i in range(int((len(space)-len(key_dname))/2)): spc=spc+' '
        tail_sapce = '' 
        for i in range((len(space)-len(key_dname))%2): tail_sapce=tail_sapce+' '
        line1 = line1 + spc + key_dname + spc + tail_sapce
    print(line1)
    for key_app in Common_Ftrs_aX_System.keys():
        spc = ''
        for j in range(int((len(space)-len(key_app))/2)): spc=spc+' '
        tail_sapce = '' 
        for j in range((len(space)-len(key_app))%2): tail_sapce=tail_sapce+' '
        line = spc + key_app + spc + tail_sapce
        for key_dname in Common_Ftrs_aX_Approach.keys():
            # val_str = '{:.2f}'.format(Prcnt_aX_Sys[key_app][key_dname])
            val_str ='{:d}/{:d}'.format(len(Common_Ftrs_aX_System[key_app]), len(Ftrs_Names[key_dname][key_app]))

            spc=''
            for k in range(int((len(space)-len(val_str))/2)): spc=spc+' '
            tail_sapce = '' 
            for k in range((len(space)-len(val_str))%2): tail_sapce=tail_sapce+' '

            line = line + '|' + spc + val_str + spc + tail_sapce
        print(line)
    print('')

    #print ax Approach#
    print('Ftr. Overlap Pct. ax Approaches ')
    line1 = space
    for key_dname in Common_Ftrs_aX_Approach.keys():
        line1 = line1 + '|' 
        spc = ''
        for i in range(int((len(space)-len(key_dname))/2)): spc=spc+' '
        tail_sapce = '' 
        for i in range((len(space)-len(key_dname))%2): tail_sapce=tail_sapce+' '
        line1 = line1 + spc + key_dname + spc + tail_sapce
    print(line1)
    for key_app in Common_Ftrs_aX_System.keys():
        spc = ''
        for j in range(int((len(space)-len(key_app))/2)): spc=spc+' '
        tail_sapce = '' 
        for j in range((len(space)-len(key_app))%2): tail_sapce=tail_sapce+' '
        line = spc + key_app + spc + tail_sapce
        for key_dname in Common_Ftrs_aX_Approach.keys():
            # val_str = '{:.2f}'.format(Prcnt_aX_App[key_app][key_dname])
            val_str = '{:d}/{:d}'.format(len(Common_Ftrs_aX_Approach[key_dname]), len(Ftrs_Names[key_dname][key_app]))
            
            spc=''
            for k in range(int((len(space)-len(val_str))/2)): spc=spc+' '
            tail_sapce = '' 
            for k in range((len(space)-len(val_str))%2): tail_sapce=tail_sapce+' '

            line = line + '|' + spc + val_str + spc + tail_sapce
        print(line)
    print('')

    return {
        'prcnt_ax_sys_mtrx': Prcnt_aX_Sys_Mtrx,
        'prcnt_ax_app_mtrx': Prcnt_aX_App_Mtrx,
        'prcnt_ax_sys': Prcnt_aX_Sys,
        'prcnt_ax_app': Prcnt_aX_App,
        'common_ftrs_sys': Common_Ftrs_aX_System,
        'common_ftrs_app': Common_Ftrs_aX_Approach,
        'ftrs_names': Ftrs_Names
    }

def Plot_LinearRegressionResult_FromFile():
    
    dataset_names = ['Cnf1.xy', 'Cnf2.xy', 'Cnf3.xy']
    methods = {
        'MAP': ['ridge', 'lasso', 'debias'], 
        'Bayes': ['ridge']
    }
    approaches = [item  for item in methods.keys()]

    Clrs={
        'MAP': {'debias':'g', 'ridge':'b', 'lasso': 'r'},
        'Bayes': {'ridge': 'm'}
    }
    Syms={
        'MAP': {'debias':'s', 'ridge':'o', 'lasso': 'D'},
        'Bayes': {'ridge': '^'}
    }
    Clrs_Sys={'Cnf1.xy':'b', 'Cnf2.xy':'r', 'Cnf3.xy':'g'}
    Syms_Sys={'Cnf1.xy':'s', 'Cnf2.xy':'v', 'Cnf3.xy':'o'}

    ## Load the entire file saving linear regression results ##
    path = './DATA/Results_data'
    Files_names = {}
    R_LinReg = {}
    for dname in dataset_names:
        Files_names[dname] = {}
        R_LinReg[dname] = {}
        for app in approaches:
            Files_names[dname][app]={}
            R_LinReg[dname][app]={}
            for reg in methods[app]:
                file_name = dname+'_'+app+'_'+reg+'_Dict'
                R_dict = dr.read_a_dictionary_file(os.path.join(path, file_name))
                Files_names[dname][app][reg] = file_name
                R_LinReg[dname][app][reg] = R_dict
    
    ## Plots the Weights to compare ##
    # - prepare the figure structure - #
    Fig, Axes = plt.subplots(3,3, sharex=True, figsize=(18,6))
    Axs={}
    for i in range(len(dataset_names)):
        dname_key = dataset_names[i]
        Axs[dname_key] = {}
        Axs[dname_key].update({'MAP-Ridge-Lasso-Debias': Axes[0,i]})
        Axs[dname_key].update({'Bayes-MAP-Ridge': Axes[1,i]})
        Axs[dname_key].update({'Bayes-MAP-Ridge-Abs': Axes[2,i]})
        Axes[0,i].set_title(dname_key,fontsize=14)
        Axes[2,i].set_xlabel(r'Feature Index $\alpha$', fontsize=12)
        if i==0:
            Axes[0,i].set_ylabel(r'$w_\alpha$', fontsize=15, labelpad=10)
            Axes[1,i].set_ylabel(r'$w_\alpha$', fontsize=15, labelpad=10)
            Axes[2,i].set_ylabel(r'$|w_\alpha|$', fontsize=15, labelpad=10)
    
    Fig_Apps = {}
    Axs_Apps = {}
    for ky1 in methods.keys():
        for ky2 in methods[ky1]:
            fg, ax = plt.subplots(1,1)
            ax.set_title(ky1+'-'+ky2, fontsize=13)
            ax.set_xlabel(r'Feature Index $\alpha$', fontsize=12)
            ax.set_ylabel(r'$w_\alpha$', fontsize=15, labelpad=-10)
            Fig_Apps[ky1+'-'+ky2] = fg
            Axs_Apps[ky1+'-'+ky2] = ax
    
    # - post treat of results & plot - #
    lines=[]
    handles=[]
    for i in range(len(dataset_names)):
        dname_key = dataset_names[i]
        ax1 = Axs[dname_key]['MAP-Ridge-Lasso-Debias']
        ax2 = Axs[dname_key]['Bayes-MAP-Ridge']
        ax3 = Axs[dname_key]['Bayes-MAP-Ridge-Abs']
        R_lr_MAP = R_LinReg[dname_key]['MAP']
        R_lr_BYA = R_LinReg[dname_key]['Bayes']
        full_features = R_lr_MAP['ridge']['feature names']

        ## MAP data prepration ##
        Map_Opt_Indexs = {}
        Map_Opt_Hypers = {}
        Map_Opt_Ws = {}
        Map_Err_Ws = {}
        Map_Ftr_Idx = {}
        for reg_key in methods['MAP']:
            Ftrs = R_lr_MAP[reg_key]['feature names']
            Rslt = R_lr_MAP[reg_key]['result']
            Pnlt = Rslt['Panelty']
            opt_idx = np.where(Rslt['Ave_Risk(panelty)']==np.min(Rslt['Ave_Risk(panelty)']))[0][0]            
            Map_Opt_Indexs[reg_key] = opt_idx
            Map_Opt_Hypers[reg_key] = Pnlt[opt_idx]
            Map_Opt_Ws[reg_key] = Rslt['Ave_Ws(panelty)'][Pnlt[opt_idx]]
            Map_Err_Ws[reg_key] = Rslt['Std_Ws(panelty)']['std'][opt_idx]
            Map_Ftr_Idx[reg_key] = np.array([full_features.index(item) for item in Ftrs])

        ## Bayes data preparation ##
        R_BYA_ridge = R_lr_BYA['ridge']['result_all_data']
        Bya_Ftr_index = np.array([full_features.index(item) for item in  R_lr_BYA['ridge']['feature names']])
        Bya_Ws = R_BYA_ridge['Ws']
        eff_num_coef = int(R_BYA_ridge['Num Eff. Ft.'][R_BYA_ridge['the_opt_idx']]+1)
        sort_ws = np.sort(np.abs(Bya_Ws))
        eff_idx = where(np.abs(Bya_Ws)>=sort_ws[-eff_num_coef])[0]
        eff_cof = np.array([ Bya_Ws[item] for item in eff_idx ])

        ## plot ws MAP Ridge, Lasso, Debias ##
        for xc in Map_Ftr_Idx['debias']:
            ax1.plot([xc, xc], [-1,10], 'k--', lw=0.4)
        ax1_ylim_min = 1e100
        ax1_ylim_max = -1e100
        for key in methods['MAP']:
            lx1 = ax1.errorbar(Map_Ftr_Idx[key], Map_Opt_Ws[key], yerr=Map_Err_Ws[key], fmt=Syms['MAP'][key], mfc='none', elinewidth=1, capsize=4, color=Clrs['MAP'][key], label='MAP-'+key)
            ax1_ylim_min = min(ax1_ylim_min, np.min(Map_Opt_Ws[key]))
            ax1_ylim_max = max(ax1_ylim_max, np.max(Map_Opt_Ws[key]))
            if i==0 :
                lines.append(lx1)
                handles.append('MAP-'+key)
            Axs_Apps['MAP-'+key].errorbar(Map_Ftr_Idx[key], Map_Opt_Ws[key], yerr=Map_Err_Ws[key], fmt=Syms_Sys[dname_key], ms=7, mfc='none', elinewidth=1, capsize=4, color=Clrs_Sys[dname_key], label=dname_key)
        ax1.set_ylim([1.1*ax1_ylim_min, 1.1*ax1_ylim_max])

        ## plot ws MAP-Bayes Reidge ##
        for xc in eff_idx:
            ax2.plot([xc, xc], [-1,10], 'k--', lw=0.4)
        key2 = 'ridge'
        ax2.errorbar(Map_Ftr_Idx[key2], Map_Opt_Ws[key2], yerr=Map_Err_Ws[key2], fmt=Syms['MAP'][key2], mfc='none', elinewidth=1, capsize=4, color=Clrs['MAP'][key2], label='MAP-'+key2)
        lbya = ax2.plot(Bya_Ftr_index, Bya_Ws, Syms['Bayes'][key2], mfc='none', color=Clrs['Bayes'][key2], label='Bayes-'+key2)
        lbay_eff = ax2.plot(eff_idx, eff_cof, 'ks', ms=8, mfc='none', mew=1, label='Bayes Eff. Ftr.')
        ax2.set_ylim([ 1.1*np.min(np.append(Bya_Ws, Map_Opt_Ws[key2])) , 1.1*np.max(np.append(Bya_Ws, Map_Opt_Ws[key2])) ])
        Axs_Apps['Bayes-ridge'].plot(Bya_Ftr_index, Bya_Ws, Syms_Sys[dname_key], ms=7, mfc='none', color=Clrs_Sys[dname_key], label=dname_key)

        if i==0:
            lines.append(lbya[0])
            handles.append('Bayes-'+key2)
            lines.append(lbay_eff[0])
            handles.append('Bayes Eff. Ftr.')
        
        ## plot |ws| MAP-Bayes Rdige
        key3 = key2
        ax3.plot(Map_Ftr_Idx[key3], np.abs(Map_Opt_Ws[key3]), Syms['MAP'][key3], mfc='none',  color=Clrs['MAP'][key3], label='MAP-'+key3)
        ax3.plot(Bya_Ftr_index, np.abs(Bya_Ws), Syms['Bayes'][key3], mfc='none', color=Clrs['Bayes'][key3], label='Bayes-'+key3)
        ax3.set_yscale('log')

    ## set legends ##
    Fig.legend(lines, handles, loc='lower center')
    for key in Axs_Apps.keys():
        Axs_Apps[key].legend()

    # print(lines)
    # print(handles)

    return Fig, Axs, lines, handles

def main_testing_GroundTruthSinX():
    pass

    gf = lambda x: np.sin(x)

    N = 400
    np.random.seed(102939028)
    x_data = np.random.uniform(0, 2*np.pi, N)
    noise = np.random.normal(0, 0.2, N)
    targets = gf(x_data) + noise
    M = 8
    features = np.array([[itm**(k+1) for k in range(M)] for itm in x_data])

    plt.figure()
    plt.title('training data')
    plt.plot(x_data, targets, 'o', mfc='none', mew=1.5)

    cross_validation_method = 'rand_cross'
    parameter_crossvalidation = {
        'validation_ratio' : 0.1, # its function depends on the opted cross-validation method. 

        ## for Nest Cross Validation ##
        'num_cell' : int(5), # number of cells in the "nest"

        ## for Segmenting Cross Validation ##
        'num_seg' : int(3), # number of segments apart from validation set

        ## for Random Split Cross Validation ##
        'num_split_ratio': 0.1,  # to generate number of trials : num_split_ratio/validation_ratio

        ## Producibility of randomness ##
        'rand_seed': 357828922 , 
        'seeding_or_not': True,   # true for reproducible cross validation data structure using 'rand_seed'

        ## Control Options ##
        'print_info_or_not': True
    }

    ### Prepare Data for Cross Validation ###
    num_trial, Data_Cluster = dxv.DataPreparation_for_CrossValidation(cross_validation_method, N, features, targets, parameter_crossvalidation)

    ### Training ###
    Penalties = np.logspace(-1,2,21)
    Betas = np.logspace(-1,2,21)
    
    ## Ridge MAP ### 
    # R_Ridge = glh_map.Model_Training(num_trial, Data_Cluster, M, regularisation='ridge', Panelty_Values=Penalties)
    # glh_map.Plot_One_Training_Result(R_Ridge)

    ### Lasso MAP ###
    # R_Lasso = glh_map.Model_Training(num_trial, Data_Cluster, M, regularisation='lasso', Panelty_Values=Penalties)
    # glh_map.Plot_One_Training_Result(R_Lasso)

    # sys.exit()

    ### Ridge Bayes ###
    R_Trials = []
    for i in range(num_trial):
        print('Trial:',i)
        fit_f = Data_Cluster[i]['fit']['feature']
        fit_t = Data_Cluster[i]['fit']['target']
        R_Trials.append(Bayes_LR_Ridge_UnNormalised_Training_SearchInFeededPenalties(Penalties, fit_t, fit_f))
    
    ### Lasso Bayes ###
    print('Lasso Bayes')
    R_Trials_Lasso = []
    for i in range(num_trial):
        print('Trail:', i)
        fit_f = Data_Cluster[i]['fit']['feature']
        fit_t = Data_Cluster[i]['fit']['target']
        R_Trials_Lasso.append(Bayes_LR_Lasso_UnNormalised_Training_SearchFeededHypes_2(Penalties, Betas,fit_t, fit_f, mc_size=1000))

    # fit_f = Data_Cluster[0]['fit']['feature']
    # fit_t = Data_Cluster[0]['fit']['target']
    # Elqs = slr.Bayes_LR_Ridge_UnNormalised_ElemtaryQuantities(fit_t, fit_f)
    # what = slr.generate_hatW_samples(Elqs['M'], 30)
    # R_xxxx = slr.Bayes_LR_Lasso_NLL(Elqs, 1, 1, what)

    # beta_test = 1.0
    # lmd_test = 1.0
    # Elqs = slr.Bayes_LR_Lasso_Un_EQ(fit_t, fit_f)
    # what = slr.generate_W_samples(Elqs['mean_ws'], Elqs['Mtrx_inv'], beta_test, 300)
    # R_xxxx = slr.Bayes_LR_Lasso_NLL_MG(Elqs, lmd_test, beta_test, what)

    # sys.exit()
    ### plot Ridge Bayes ###
    clrs = cm.rainbow(np.linspace(0,1,num_trial))
    plt.figure()
    plt.title('Ridge Bayes Nlls')
    for i in range(num_trial):
        plt.plot(Penalties, R_Trials[i]['nlls'], 'o--', lw=1, mew=1, ms=8, color=clrs[i], mfc='none')
        # plt.plot(penalties, R_Maps[i]['emp_risk'], '^--', lw=1, mew=1, ms=8, color=clrs[i], mfc='none')
    plt.xscale('log')
    plt.legend()

    plt.figure()
    plt.title('Ridge Bayes Ws')
    for i in range(num_trial):
        plt.plot(R_Trials[i]['Ws'], 'o--', lw=1, mew=1, ms=8, color=clrs[i], mfc='none')
        # plt.plot(penalties, R_Maps[i]['emp_risk'], '^--', lw=1, mew=1, ms=8, color=clrs[i], mfc='none')
    plt.legend()

    plt.figure()
    plt.title('Ridge Bayes Betas')
    for i in range(num_trial):
        plt.plot(Penalties, R_Trials[i]['betas'], 's--', lw=1, mew=1.5, ms=8, color=clrs[i], mfc='none')
    plt.legend()

    # sys.exit()
    ### plot Lasso Bayes ###
    # print('plot Lasso')
    # clrs = cm.rainbow(np.linspace(0,1,num_trial))
    # for i in range(num_trial):
    #     plt.figure()
    #     plt.title('Lasso Bayes Nlls, trial: {:d}'.format(i) )
    #     plt.imshow(R_Trials_Lasso[i]['R_Matrix']['nll'])
    #     plt.colorbar()

    plt.figure()
    plt.title('Lasso Bayes Ws')
    for i in range(num_trial):
        plt.plot(R_Trials_Lasso[i]['Ave_Ws'], 'o--', lw=1, mew=1, ms=8, color=clrs[i], mfc='none')
        # plt.plot(penalties, R_Maps[i]['emp_risk'], '^--', lw=1, mew=1, ms=8, color=clrs[i], mfc='none')
    plt.legend()

    # sys.exit()

def main_FullyFunctional_BayesianRidge(): # !! be causcious, may overwrite file, commend out the saving code to avoid overwritting !! 
    pass 

    ### Global Parameters ###
    data_std_cut = 1e-4 # for neglecting non-varying components in feature data - dr.load_data(..., std_cut=data_std_cut, ...) 

    ### Data Loading & pre-Processing ###
    home_path = '/Users/chenliu/'
    project_path = 'Research_Projects/SVM-SwapMC'
    training_data_path = 'DATA/Training_data'
    training_data_file = 'Cnf3.xy'
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
        'validation_ratio' : 0.1, # its function depends on the opted cross-validation method. 

        ## for Nest Cross Validation ##
        'num_cell' : int(5), # number of cells in the "nest"

        ## for Segmenting Cross Validation ##
        'num_seg' : int(3), # number of segments apart from validation set

        ## for Random Split Cross Validation ##
        'num_split_ratio': 0.5,  # to generate number of trials : num_split_ratio/validation_ratio

        ## Producibility of randomness ##
        'rand_seed': 357828922 , 
        'seeding_or_not': True,   # true for reproducible cross validation data structure using 'rand_seed'

        ## Control Options ##
        'print_info_or_not': True
    }

    ### Prepare Data for Cross Validation ###
    num_trial, Data_Cluster = dxv.DataPreparation_for_CrossValidation(cross_validation_method, N, features, targets, parameter_crossvalidation)

    ### training part ###
    penalties = np.logspace(-2,8,21)
    Betas = np.logspace(-1,2,11)

    ### Bayes Ridge ###
    print('Training Bayes Ridge ...')
    R_Ridge_WithAllData = Bayes_LR_Ridge_UnNormalised_Training_SearchInFeededPenalties(penalties, targets, features)
    R_Trials_Ridge = []
    for i in range(num_trial):
        print('Trial:',i)
        fit_f = Data_Cluster[i]['fit']['feature']
        fit_t = Data_Cluster[i]['fit']['target']
        R_Trials_Ridge.append(Bayes_LR_Ridge_UnNormalised_Training_SearchInFeededPenalties(penalties, fit_t, fit_f))

    ### Save results to file ###
    R_save = {'feature names': feature_names, 'result_all_data': R_Ridge_WithAllData, 'result_trials': R_Trials_Ridge}
    # dr.save_to_file_a_dictrionary('./DATA/Results_data/'+training_data_file+'_Bayes_ridge_Dict', R_save)

    # sys.exit()
    ### Plot Bayes Ridge ###
    ## plot model evidence ##
    clrs = cm.rainbow(np.linspace(0,1,num_trial))
    fig_nll, ax_nll = plt.subplots()
    ax_nll_zoom = fig_nll.add_axes([0.39,0.36,0.49,0.49])
    ax_nll.set_title(r'$-1 \times \ln($ Model Evidence $)$ '+training_data_file)
    # Av_nll=np.zeros(len(penalties))
    for i in range(num_trial):
        ax_nll.plot(penalties, R_Trials_Ridge[i]['nlls'], 'o--', lw=1, mew=1, ms=8, color=clrs[i], mfc='none', label='trial.{:d}'.format(i+1))
        ax_nll_zoom.plot(penalties, R_Trials_Ridge[i]['nlls'], 'o--', lw=1, mew=1, ms=8, color=clrs[i], mfc='none')
        # plt.plot(penalties, R_Maps[i]['emp_risk'], '^--', lw=1, mew=1, ms=8, color=clrs[i], mfc='none')
        # Av_nll = Av_nll + R_Trials_Ridge[i]['nlls']*(1.0/num_trial)
    ax_nll.plot(penalties, R_Ridge_WithAllData['nlls'], '^--', lw=1, mew=1, ms=6, color='k', mfc='k', label='All data')
    ax_nll_zoom.plot(penalties, R_Ridge_WithAllData['nlls'], '^--', lw=1, mew=1, ms=6, color='k', mfc='k')
    # plt.plot(penalties, Av_nll, 's--', lw=1, mew=1, ms=8, color='k', mfc='none')
    ax_nll.set_xscale('log')
    ax_nll_zoom.set_xscale('log')
    ax_nll_zoom.set_xlim([24000,2.74e7])
    ax_nll_zoom.set_ylim([0.785,0.832])
    ax_nll.legend(loc=3)
    ax_nll.set_xlabel(r'$\tilde\lambda$', fontsize=16, loc='right', labelpad=-12)

    ## plot weights ##
    lmd_ref=100
    idx_ref = where(np.abs(penalties-lmd_ref)==min(np.abs(penalties-lmd_ref)))[0][0]
    Ref_Ws_Trials = []
    Opt_Ws_Trials = []
    for i in range(num_trial):
        Ref_Ws_Trials.append(R_Trials_Ridge[i]['All_Ws'][idx_ref])
        Opt_Ws_Trials.append(R_Trials_Ridge[i]['Ws'])
    Ref_Ws_Trials = np.array(Ref_Ws_Trials)
    Opt_Ws_Trials = np.array(Opt_Ws_Trials)
    
    Av_Ref_Ws_Trials = np.zeros(M)
    Std_Ref_Ws_Trials = np.zeros(M)
    Av_Opt_Ws_Trials = np.zeros(M)
    Std_Opt_Ws_Trials = np.zeros(M)

    for k in range(M):
        Av_Ref_Ws_Trials[k] = np.mean(Ref_Ws_Trials[:,k])
        Std_Ref_Ws_Trials[k] = np.std(Ref_Ws_Trials[:,k])
        Av_Opt_Ws_Trials[k] = np.mean(Opt_Ws_Trials[:,k])
        Std_Opt_Ws_Trials[k] = np.std(Opt_Ws_Trials[:,k])
    
    Opt_Ws_AllData = R_Ridge_WithAllData['Ws']

    eff_num_coef = int(R_Ridge_WithAllData['Num Eff. Ft.'][R_Ridge_WithAllData['the_opt_idx']]+1)
    sort_ws = np.sort(np.abs(Av_Opt_Ws_Trials))
    eff_idx = where(np.abs(Av_Opt_Ws_Trials)>=sort_ws[-eff_num_coef])[0]
    eff_cof = []
    for item in eff_idx:
        eff_cof.append(Av_Opt_Ws_Trials[item])
    eff_cof = np.array(eff_cof)

    fig_ws, (ax1, ax2) = plt.subplots(2,1,sharex=True)
    ax1_ref = ax1.twinx()
    ax1.set_title(r'$w_\alpha$ '+training_data_file, fontsize=15)
    ax1_ref.set_ylim([-0.067,0.067])
    ax1.set_ylim([-0.001, 0.001])
    ax1.set_ylabel('Bayes', loc='top', rotation=0, labelpad=-50, fontsize=12)
    ax1_ref.set_ylabel('MAP', loc='top', rotation=0, labelpad=-10, fontsize=12)
    # ax1_ref.errorbar(np.arange(M), Av_Ref_Ws_Trials, yerr=Std_Ref_Ws_Trials, fmt='o', mfc='none', elinewidth=1, capsize=4, color='b', label='MAP Ridge')
    ax1.errorbar(np.arange(M), Av_Opt_Ws_Trials, yerr=Std_Opt_Ws_Trials, fmt='d', mfc='none', elinewidth=1, capsize=4, color='c', label='Bayes Trials')
    ax1.plot(eff_idx, eff_cof, 'ks', ms=8, mfc='none', mew=1)
    ax1.plot(np.arange(M), Opt_Ws_AllData, 'm^', mfc='none', ms=7, label='Bayes All data')
    ax1.legend(loc=1)
    ax1_ref.legend(loc=4)
    ax2.set_title(r'$|w_\alpha|$', fontsize=15)
    ax2.set_xlabel(r'Features Index $\alpha$')
    # ax2.set_ylabel('Abs')
    ax2.plot(np.arange(M), Opt_Ws_AllData, 'm^', mfc='m', ms=7, label='Bayes: +')
    ax2.plot(np.arange(M), np.abs(Opt_Ws_AllData), 'm^', mfc='none', ms=7, label='Bayes: -')
    # ax2.plot(np.arange(M), Av_Ref_Ws_Trials, 'bo', mfc='b', ms=7, label='MAP: +')
    # ax2.plot(np.arange(M), np.abs(Av_Ref_Ws_Trials), 'bo', mfc='none', ms=7, label='MAP: -')
    ax2.plot(eff_idx, np.abs(eff_cof), 'ks', ms=8, mfc='none', mew=1)
    ax2.set_yscale('log')
    ax2.legend()

    ## plot Ws covariance ##
    plt.figure()
    plt.title(r'Covariance Map of $w_\alpha$ '+training_data_file, fontsize=14)
    plt.imshow(R_Ridge_WithAllData['CVar_Ws'])
    plt.colorbar()


if __name__ == '__main__':
    
    print('GaussianLH_Penalty_Ridge_Bayes.py')
    sys.exit()

    main_FullyFunctional_BayesianRidge()
    sys.exit()

    # main_testing_GroundTruthSinX()
    # sys.exit()