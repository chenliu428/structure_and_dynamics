import numpy as np
import math as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc, rcParams
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
from scipy import fftpack as ftp

import sys
from datetime import datetime
import os.path
import imp

import data_read as dr 

imp.reload(dr)

### Tools for plotting MAP regression results ###
def Plot_Several_Training_Results(Results_Dict:dict, Features_Idexs_In:dict={}, Data_name:str='X', Symbos:list=['o','D', 's', '>', 'x','H','^'], Colors=['b', 'r', 'g', 'm', 'c','k']):
    if len(Results_Dict.keys())>len(Symbos) or len(Results_Dict.keys())>len(Colors):
        print("no sufficient symbos or colors")
        sys.exit()

    Features_Idexs = {}
    Optimal_Indexes = {}
    Optimal_Hypers = {}
    Optimal_Ws_Ave = {}
    Optimal_Ws_Std = {}
    Syms={}
    Clrs={}
    it=0
    for key in Results_Dict.keys():
        Syms[key] = Symbos[it]
        Clrs[key] = Colors[it]
        Rslt = Results_Dict[key]
        Pnlt = Results_Dict[key]['Panelty']
        opt_idx = np.where(Rslt['Ave_Risk(panelty)']==np.min(Rslt['Ave_Risk(panelty)']))[0][0]
        Optimal_Indexes[key] = opt_idx
        Optimal_Hypers[key] = Pnlt[opt_idx]
        Optimal_Ws_Ave[key] = Rslt['Ave_Ws(panelty)'][Pnlt[opt_idx]]
        Optimal_Ws_Std[key] = Rslt['Std_Ws(panelty)']['std'][opt_idx]
        if key not in Features_Idexs_In.keys():
            Features_Idexs[key] = np.arange(len(Rslt['Ave_Ws(panelty)'][Pnlt[opt_idx]]))
        else:
            Features_Idexs[key] = Features_Idexs_In[key]
        it=it+1

    print('optimal hyper-parameters: ')
    for key in Results_Dict.keys():
        print(str(key)+' : idx = {:d}, value = {:.2E}'.format(Optimal_Indexes[key], Optimal_Hypers[key]))
    
    fig_risk, ax_risk = plt.subplots()
    ax_risk.set_xlabel(r'$\tilde{\lambda}$', fontsize=15)
    ax_risk.set_title('Expected loss per data point '+Data_name, fontsize=15)
    ax_risk_zoom = fig_risk.add_axes([0.39,0.26,0.47,0.49])
    ax_risk_zoom.set_xlabel(r'$\tilde{\lambda}$')
    ax_risk_zoom.set_xlim([4.1e-7,47])
    ax_risk_zoom.set_ylim([0.233,0.246])
    ax_risk_zoom.set_xscale('log')

    fig_nll, ax_nll = plt.subplots()
    ax_nll.set_xlabel(r'$\tilde{\lambda}$', fontsize=15)
    ax_nll.set_title('Expected NLL '+Data_name, fontsize=15)
    ax_nll_zoom = fig_nll.add_axes([0.39,0.26,0.47,0.49])
    ax_nll_zoom.set_xlabel(r'$\tilde{\lambda}$')
    ax_nll_zoom.set_xlim([4.1e-7,47])
    ax_nll_zoom.set_ylim([0.68,0.71])
    ax_nll_zoom.set_xscale('log')

    fig_risk_nll_mix, ax_mix_risk = plt.subplots()
    ax_mix_nll = ax_mix_risk.twinx()
    ax_mix_risk.set_title(r'Expected Loss & Nll '+Data_name )
    ax_mix_risk.set_xlabel(r'$\tilde{\lambda}$', fontsize=15)
    ax_mix_risk.set_ylabel(r'Risk', labelpad=-5, loc='top', fontsize=15, rotation=0)
    ax_mix_nll.set_ylabel(r'Nll', labelpad=-5, loc='top', fontsize=15, rotation=0)
    
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    ax1.set_title(Data_name, fontsize=15)
    ax1.set_ylabel(r'$W_\alpha$')
    ax2.set_ylabel(r'$|W_\alpha|$')
    ax2.set_xlabel(r'Features Index $\alpha$')
    ax2.set_yscale('log')

    Min_Feature_key = '???'
    N_ft_min = 10000000000
    for key in Features_Idexs.keys():
        if len(Features_Idexs[key])<N_ft_min: 
            N_ft_min = len(Features_Idexs[key])
            Min_Feature_key = key
    for xc in Features_Idexs[Min_Feature_key]:
        ax1.plot([xc, xc], [-1,10], 'k--', lw=0.4)
        ax2.plot([xc, xc], [-1,10], 'k--', lw=0.4)

    for key in Results_Dict.keys():
        Rslt = Results_Dict[key]
        
        ax_risk.plot(Rslt['Panelty'], Rslt['Ave_Risk(panelty)'], Clrs[key]+Syms[key]+'--', ms=8, mfc='none', mew=1, lw=1, label=str(key))
        ax_risk_zoom.plot(Rslt['Panelty'], Rslt['Ave_Risk(panelty)'], Clrs[key]+Syms[key]+'--', ms=8, mfc='none', mew=1, lw=1, label=str(key))
        ax_nll.plot(Rslt['Panelty'], Rslt['Ave_Nll(panelty)'], Clrs[key]+Syms[key]+'--', ms=8, mfc='none', mew=1, lw=1, label=str(key))
        ax_nll_zoom.plot(Rslt['Panelty'], Rslt['Ave_Nll(panelty)'], Clrs[key]+Syms[key]+'--', ms=8, mfc='none', mew=1, lw=1, label=str(key))

        if key!='lasso':
            ax_mix_risk.plot(Rslt['Panelty'], Rslt['Ave_Risk(panelty)'], Clrs[key]+Syms[key]+'--', ms=8, mfc='none', mew=1, lw=1, label=str(key)+' risk')
            ax_mix_nll.plot(Rslt['Panelty'], Rslt['Ave_Nll(panelty)'], Clrs[key]+Syms[key]+'-.', ms=8, mfc=Clrs[key], mew=1, lw=1, label=str(key)+' nll')

        ax1.errorbar(Features_Idexs[key], Optimal_Ws_Ave[key], yerr=Optimal_Ws_Std[key], fmt=Syms[key], mfc='none', elinewidth=1, capsize=4, color=Clrs[key], label=str(key))
        # ax2.errorbar(np.arange(len(Ws_Ave_Ridge)), np.abs(Ws_Ave_Ridge), yerr=Ws_Std_Ridge, fmt='o', mfc='none', elinewidth=1, capsize=4, color='b', label='Ridge')
        ax2.plot(Features_Idexs[key], np.abs(Optimal_Ws_Ave[key]), Syms[key], mfc='none', color=Clrs[key], label=str(key))
    
    fig_risk_nll_mix.legend(loc='upper right', bbox_to_anchor=(0.52, 0.87))

    ax_risk.legend(loc=2)    
    ax_nll.legend(loc=2)

    ax1.set_ylim([-0.064,0.067])
    ax2.set_ylim([4e-7,0.1])
    ax1.legend()
    
    return {
        'fig_risk': fig_risk, 
        'ax_risk': ax_risk, 
        'ax_risk_zoom': ax_risk_zoom,
        'fig_nll': fig_nll, 
        'ax_nll': ax_nll, 
        'ax_nll_zoom': ax_nll_zoom, 
        'fig_risk_nll_mix': fig_risk_nll_mix, 
        'ax_mix_risk': ax_mix_risk, 
        'ax_mix_nll': ax_mix_nll, 
        'fig_ws':fig, 
        'ax_ws_1': ax1, 
        'ax_Ws_2': ax2, 
        'symbols': Syms, 
        'colors': Clrs
    }

def Plot_One_Training_Result(R_Ridge):
    pnlt_ridge = R_Ridge['Panelty']

    optimal_idx_ridge = np.where(R_Ridge['Ave_Risk(panelty)']==np.min(R_Ridge['Ave_Risk(panelty)']))[0][0]
    print('optimal idx: ', optimal_idx_ridge)

    Ws_Ave_Ridge = R_Ridge['Ave_Ws(panelty)'][pnlt_ridge[optimal_idx_ridge]]
    Ws_Std_Ridge = R_Ridge['Std_Ws(panelty)']['std'][optimal_idx_ridge,:]

    fig_risk, ax_risk = plt.subplots()
    ax_risk.plot(R_Ridge['Panelty'], R_Ridge['Ave_Risk(panelty)'], 'bo--', ms=8, mfc='none', mew=1, lw=1, label='Ridge')
    ax_risk.set_xlabel(r'$\tilde{\lambda}$', fontsize=15)
    ax_risk.set_title('Expected loss per data point', fontsize=15)
    ax_risk.legend()
    # ax_risk_zoom = fig_risk.add_axes([0.39,0.26,0.47,0.49])
    # ax_risk_zoom.plot(R_Ridge['Panelty'], R_Ridge['Ave_Risk(panelty)'], 'bo--', ms=8, mfc='none', mew=1, lw=1, label='Ridge')
    # ax_risk_zoom.set_xlim([30,210])
    # ax_risk_zoom.set_ylim([0.231,0.23158])
    # ax_risk_zoom.set_xlabel(r'$\tilde{\lambda}$')

    fig_nll, ax_nll = plt.subplots()
    ax_nll.plot(R_Ridge['Panelty'], R_Ridge['Ave_Nll(panelty)'], 'bo--', ms=8, mfc='none', mew=1, lw=1, label='Ridge')
    ax_nll.set_xlabel(r'$\tilde{\lambda}$', fontsize=15)
    ax_nll.set_title('Expected NLL', fontsize=15)
    ax_nll.legend()
    # ax_nll_zoom = fig_nll.add_axes([0.39,0.26,0.47,0.49])
    # ax_nll_zoom.plot(R_Ridge['Panelty'], R_Ridge['Ave_Nll(panelty)'], 'bo--', ms=8, mfc='none', mew=1, lw=1, label='Ridge')
    # ax_nll_zoom.set_xlim([30,210])
    # ax_nll_zoom.set_ylim([0.682,0.6836])
    # ax_nll_zoom.set_xlabel(r'$\tilde{\lambda}$')

    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_ylabel(r'$W_\alpha$')
    ax2.set_ylabel(r'$|W_\alpha|$')
    ax2.set_xlabel(r'Features Index $\alpha$')
    ax1.errorbar(np.arange(len(Ws_Ave_Ridge)), Ws_Ave_Ridge, yerr=Ws_Std_Ridge, fmt='o', mfc='none', elinewidth=1, capsize=4, color='b', label='Ridge')
    # ax2.errorbar(np.arange(len(Ws_Ave_Ridge)), np.abs(Ws_Ave_Ridge), yerr=Ws_Std_Ridge, fmt='o', mfc='none', elinewidth=1, capsize=4, color='b', label='Ridge')
    ax2.plot(np.arange(len(Ws_Ave_Ridge)), np.abs(Ws_Ave_Ridge), 'o', mfc='none', color='b', label='Ridge')
    # ax1.set_ylim([-0.053,0.065])
    # ax2.set_ylim([3.8e-7,0.13])
    ax2.set_yscale('log')
    ax1.legend()


### Tools for plotting Bayesian Regression Results ###
def Plot_Bayes_Result(Cnf_name:str, method:str='ridge', path:str='./DATA/Results_data'):

    file_name = Cnf_name+'_Bayes_'+method+'_Dict'
    Rdict = dr.read_a_dictionary_file(os.path.join(path, file_name))

    R_Trials_Ridge = Rdict['result_trials']
    R_Ridge_WithAllData = Rdict['result_all_data']
    feature_names = Rdict['feature names']

    penalties = R_Ridge_WithAllData['Penalties']
    num_trial = len(R_Trials_Ridge)
    M = len(feature_names)

    ## plot model evidence ##
    clrs = cm.rainbow(np.linspace(0,1,num_trial))
    fig_nll, ax_nll = plt.subplots()
    ax_nll_zoom = fig_nll.add_axes([0.39,0.36,0.49,0.49])
    ax_nll.set_title(r'$-1 \times \ln($ Model Evidence $)$ '+Cnf_name)
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
    idx_ref = np.where(np.abs(penalties-lmd_ref)==min(np.abs(penalties-lmd_ref)))[0][0]
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
    eff_idx = np.where(np.abs(Av_Opt_Ws_Trials)>=sort_ws[-eff_num_coef])[0]
    eff_cof = []
    for item in eff_idx:
        eff_cof.append(Av_Opt_Ws_Trials[item])
    eff_cof = np.array(eff_cof)

    fig_ws, (ax1, ax2) = plt.subplots(2,1,sharex=True)
    # ax1_ref = ax1.twinx()
    ax1.set_title(r'$w_\alpha$ '+Cnf_name, fontsize=15)
    # ax1_ref.set_ylim([-0.067,0.067])
    ax1.set_ylim([-0.001, 0.001])
    ax1.set_ylabel('Bayes', loc='top', rotation=0, labelpad=-50, fontsize=12)
    # ax1_ref.set_ylabel('MAP', loc='top', rotation=0, labelpad=-10, fontsize=12)
    # ax1_ref.errorbar(np.arange(M), Av_Ref_Ws_Trials, yerr=Std_Ref_Ws_Trials, fmt='o', mfc='none', elinewidth=1, capsize=4, color='b', label='MAP Ridge')
    ax1.errorbar(np.arange(M), Av_Opt_Ws_Trials, yerr=Std_Opt_Ws_Trials, fmt='d', mfc='none', elinewidth=1, capsize=4, color='c', label='Bayes Trials')
    ax1.plot(eff_idx, eff_cof, 'ks', ms=8, mfc='none', mew=1)
    ax1.plot(np.arange(M), Opt_Ws_AllData, 'm^', mfc='none', ms=7, label='Bayes All data')
    ax1.legend(loc=1)
    # ax1_ref.legend(loc=4)
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
    plt.title(r'Covariance Map of $w_\alpha$ '+Cnf_name, fontsize=14)
    plt.imshow(R_Ridge_WithAllData['CVar_Ws'])
    plt.colorbar()

    return {
        'fig_nll': fig_nll,
        'ax_nll': ax_nll,
        'ax_nll_zoom': ax_nll_zoom,
        'fig_ws': fig_ws,
        'ax_ws_1': ax1,
        'ax_ws_2': ax2
    }
