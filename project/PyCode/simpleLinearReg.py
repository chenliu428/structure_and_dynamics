import math as ma
from numpy import *
from pylab import *
from scipy import *
import os.path
from matplotlib import rc, rcParams
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab

from scipy import fftpack as ftp

from scipy import *
from scipy.optimize import curve_fit, leastsq
import scipy.linalg as scplin

from scipy import interpolate as interplt
from scipy.interpolate import interp1d
import sys
from datetime import datetime
import os.path
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import imp

def convertor_base10(X): # find the scientific notation for 'exp(X)'
    ln10 = log(10.0)
    pwr = int(floor(X/ln10))
    lnx = X - pwr*ln10
    x = exp(lnx)
    return {'coefficient':x, 'exponent':pwr}

def Norm_HD(x):
    # print(len(x), ' , ', np.sum(x*x) )
    r = (np.sum(x*x)/len(x))**0.5
    return r

# def Bayes_LR_Ridge_betaW0Hyper_ElemtaryQuantities(targets, features, print_info: bool=False, plot_info: bool=True):
#     N = shape(targets)[0]
#     N_f = shape(features)[0]
#     M = shape(features)[1]

#     if N!=N_f :
#         print('N_targets != N_features, number of data points error!')
#         sys.exit()

#     ave_target = np.mean(targets)
#     ave_square_target = np.mean(targets**2)
#     b0 = np.zeros(M)
#     b1 = np.zeros(M)
#     for i in range(M):
#         b0[i] = np.sum(targets*features[:,i])
#         b1[i] = -np.sum(features[:,i])
#     Mtrx = np.zeros([M,M])
#     for i in range(M):
#         for j in range(M):
#             Mtrx[i,j] = np.sum(features[:,i]*features[:,j])

#     return {'Mtrx': Mtrx, 'b0': b0, 'b1':b1, 'ave_y': ave_target, 'ave_y2': ave_square_target, 'N': N, 'M': M}

#g
def Bayes_LR_Ridge_UnNormalised_ElemtaryQuantities(targets, features, print_info: bool=False, plot_info: bool=True):
    N = shape(targets)[0]
    N_f = shape(features)[0]
    M = shape(features)[1]

    if N!=N_f :
        print('N_targets != N_features, number of data points error!')
        sys.exit()

    ave_target = np.mean(targets)
    ave_feature = np.array([np.mean(features[:,i]) for i in range(M)])

    centered_targets = targets - ave_target
    centered_features = copy(features)
    for i in range(M):
        centered_features[:,i] = features[:,i] - np.mean(features[:,i])
    B_vct = np.zeros(M)
    for i in range(M):
        B_vct[i] = np.sum(centered_targets*centered_features[:,i])
    Mtrx = np.zeros([M,M])
    for i in range(M):
        for j in range(M):
            Mtrx[i,j] = np.sum(centered_features[:,i]*centered_features[:,j])

    return {'Mtrx': Mtrx, 'B': B_vct, 'Y': centered_targets, 'X': centered_features, 'ave_y': ave_target, 'ave_feature': ave_feature, 'N': N, 'M': M}
#g
def Bayes_LR_Lasso_Un_EQ(targets, features, print_info:bool=False, plot_info:bool=False):
    Eqs = Bayes_LR_Ridge_UnNormalised_ElemtaryQuantities(targets, features)
    Mtrx = Eqs['Mtrx']
    B_vct = Eqs['B']
    M = Eqs['M']

    log_Mtrx_norm = 0 
    for i in range(M):
        log_Mtrx_norm = log_Mtrx_norm + log(abs(Mtrx[i,i]))
    av_log_Mtrx_norm = log_Mtrx_norm/M 
    Mnorm = exp(av_log_Mtrx_norm)
    Mtrx_rn = Mtrx/Mnorm 
    det_Mtrx_rn = np.linalg.det(Mtrx_rn)
    log_det_Mtrx = M*log(Mnorm) + log(det_Mtrx_rn)

    Mtrx_inv_bis = scplin.inv(Mtrx_rn)/Mnorm
    Mtrx_inv_bis2 = scplin.inv(Mtrx)

    Ev, U = np.linalg.eigh(Mtrx)
    Di_Mtrx = np.zeros(shape(Mtrx))
    for i in range(len(Mtrx)):
        Di_Mtrx[i,i] = 1.0 / Ev[i]
    Mtrx_inv = U @ Di_Mtrx @ U.T

    if not plot_info:
        plt.figure()
        plt.title('U * UT')
        plt.imshow(U @ U.T)
        plt.colorbar()
        
        plt.figure()
        plt.title('UT * U')
        plt.imshow(U.T @ U)
        plt.colorbar()
        
        plt.figure()
        plt.title('M * M^-1')
        plt.imshow(Mtrx @ Mtrx_inv)
        plt.colorbar()

        plt.figure()
        plt.title('M^-1 * M')
        plt.imshow(Mtrx_inv @ Mtrx)
        plt.colorbar()

    mean_vct = np.dot(Mtrx_inv, B_vct)

    Eqs.update( {'mean_ws': mean_vct, 'log_det_M': log_det_Mtrx, 'Mtrx_inv': Mtrx_inv, 'Mtrx_inv_bis':Mtrx_inv_bis, 'Ev_Mtrx': Ev, 'U': U} )

    return Eqs
#g
def K_related_UnNormalised(ElementaryQuantities, tilde_lambda, print_info: bool=False, plot_info: bool=False):
    N = ElementaryQuantities['N']
    M = ElementaryQuantities['M']
    Mtrx = ElementaryQuantities['Mtrx']
    B_vct = ElementaryQuantities['B']
    ave_target = ElementaryQuantities['ave_y']
    ave_feature = ElementaryQuantities['ave_feature']
    Y = ElementaryQuantities['Y']

    K_Mtrx = Mtrx + tilde_lambda*np.eye(M)
    # K_Mtrx_inv2 = np.linalg.inv(K_Mtrx)

    Ev, U = np.linalg.eigh(K_Mtrx)
    Di_KMtrx = np.zeros(shape(K_Mtrx))
    for i in range(len(Mtrx)):
        Di_KMtrx[i,i] = 1.0 / Ev[i]
    K_Mtrx_inv = U @ Di_KMtrx @ U.T

    if plot_info:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,5))
        ax1.set_title('M * M^-1')
        im1 = ax1.imshow(np.dot(K_Mtrx, K_Mtrx_inv))
        plt.colorbar(im1, ax=ax1)
        ax2.set_title('M^-1 * M')
        im2 = ax2.imshow(np.dot(K_Mtrx_inv, K_Mtrx))
        plt.colorbar(im2, ax=ax2)

    # print('det 1 : ')
    det_K = 1 #np.linalg.det(K_Mtrx)

    log_Knorm = 0
    for i in range(M):
        log_Knorm = log_Knorm + log(abs(K_Mtrx[i,i]))
    av_log_Knorm = log_Knorm / M
    Knorm = exp(av_log_Knorm)
    K_rn = K_Mtrx/Knorm

    # K_Mtrx_inv_bis = (np.linalg.inv(K_rn))/Knorm

    # print('det 2 : ')
    det_K_rn = np.linalg.det(K_rn)
    log_det_K = M*log(Knorm) + log(det_K_rn)

    Coeff_C = np.sum(Y*Y) - np.dot(B_vct, np.dot(K_Mtrx_inv, B_vct))

    beta_opt = float(N)/Coeff_C

    ell_1 = -M*log(tilde_lambda)/float(2*N) 
    ell_2 = log_det_K/float(2*N) 
    ell_2_bis = log(det_K)/float(2*N) 
    ell_3 = 0.5*(1-1.0/float(N))*log(Coeff_C/float(N-1))
    ell_4 = (0.5+0.5*log(2*np.pi))*(1.0-1.0/float(N)) + log(N)/float(2*N)

    nll = ell_1+ell_2+ell_3+ell_4
    nll_bis = ell_1+ell_2_bis+ell_3+ell_4

    Ave_Ws = np.dot(K_Mtrx_inv, B_vct)
    Ave_W0 = ave_target - np.dot(Ave_Ws, ave_feature)

    Var_W0 = 1.0/(beta_opt*N) + np.dot(ave_feature, np.dot(K_Mtrx_inv, ave_feature))/beta_opt
    CVar_Ws = K_Mtrx_inv/beta_opt

    Num_Eff_Features = np.sum(Ave_Ws**2)*beta_opt*tilde_lambda

    return {'K': K_Mtrx, 'K_inv': K_Mtrx_inv, 'C': Coeff_C, 'det_K': det_K, 'det_K_rn': det_K_rn, 'log_det_K': log_det_K, 'l1':ell_1, 'l2':ell_2, 'l2_bis': ell_2_bis, 'l3': ell_3, 'Knorm':Knorm, 'std_Krn': np.std(K_rn), 'nll': nll, 'nll_bis': nll_bis, 'beta': beta_opt, 'Ave_W0': Ave_W0, 'Ave_Ws': Ave_Ws, 'Var_W0': Var_W0, 'CVar_Ws': CVar_Ws, 'Num Eff. Ft.': Num_Eff_Features}
#g
def generate_hatW_samples(M, Num_Samples, seeding:bool=True, rand_seed=364632321):
    if seeding:
        np.random.seed(rand_seed)
    nor = [np.random.exponential(1, M) for i in range(Num_Samples)]
    sig = [np.sign(np.random.uniform(-1,1, M)) for i in range(Num_Samples)]

    return [nor[i]*sig[i] for i in range(Num_Samples)]
#g
def generate_W_samples(mean, mtrx_inv, beta, Num_Samples, seeding:bool=True, rand_seed=364632321):
    if seeding:
        np.random.seed(rand_seed)
    x = np.random.multivariate_normal(mean=mean, cov=mtrx_inv/beta, size=Num_Samples)
    return [x[i,:] for i in range(Num_Samples)]

# def generate_W_samples_via_Utransform(mean, U, Ev_of_Mtrx, beta, Num_Samples, seeding:bool=True, rand_seed=364632321):
#     if seeding:
#         np.random.seed(rand_seed)
#     CovD = np.zeros(shape(U))
#     for i in range(len(Ev_of_Mtrx)):
#         CovD[i,i] = 1.0/Ev_of_Mtrx[i]/beta
#     Ys = np.random.multivariate_normal(mean=np.zeros(len(mean)), cov=CovD, size=Num_Samples)
    
#     return [np.dot(U, Ys[i,:]) + mean for i in range(Num_Samples)]

# def func_low(x, p, c) : 
#     return p*x + c

# def func_high(x, p, a, c):
#     return a*(x**p) + c

# def fit_gGamma_dist_sum_absWs(w_samples, bin_ratio=0.005, low_interval_cut_ratio = 0.1, high_interval_cut_ratio = 10.0):
#     sample_size = len(w_samples)
#     bin_num = int(sample_size*bin_ratio)
#     ells = np.array([ np.sum(np.abs(item)) for item in w_samples])
#     yh, xh = np.histogram(ells, density=1, bins=np.logspace(np.log10
#     (min(ells)), np.log10(max(ells)), bin_num))
#     x = 0.5*(xh[:-1]+xh[1:])
#     mean_ell = np.mean(ells)
#     low_x = []
#     low_y = []
#     high_x = []
#     high_y = []
#     for i in range(len(x)):
#         if x[i]<mean_ell*low_interval_cut_ratio :
#             low_x.append(x[i])
#             low_y.append(yh[i])
#         if x[i]>mean_ell*high_interval_cut_ratio :
#             high_x.append(x[i])
#             high_y.append(yh[i])
#     low_x = np.array(low_x)
#     low_y = np.array(low_y)
#     high_x = np.array(high_x)
#     high_y = np.array(high_y)

#     log_low_x = np.log(low_x)
#     log_low_y = np.log(low_y)
#     log_high_y = np.log(high_y)

#     par_l, pav_l = curve_fit(func_low, log_low_x, log_low_y)
#     par_h, pav_h = curve_fit(func_high, high_x, log_high_y)

#     plt.figure()
#     plt.plot(x, yh, 'ro', ms=8, mew=1, mfc='none', mec='r')
#     plt.plot(low_x, low_x**par_l[0]*par_l[1], 'b-', lw=1)
#     plt.plot(high_x, np.exp(func_high(high_x, par_h)), 'g-', lw=1)

#g
def Bayes_LR_Lasso_NLL(ElementaryQuantities, tilde_lambda, beta, hatW_samples:list, redo_sampling:bool=False, redo_sampling_size:int=1000):
    N = ElementaryQuantities['N']
    M = ElementaryQuantities['M']
    Mtrx = ElementaryQuantities['Mtrx']
    B_vct = ElementaryQuantities['B']
    ave_target = ElementaryQuantities['ave_y']
    ave_feature = ElementaryQuantities['ave_feature']
    Y = ElementaryQuantities['Y']

    if not redo_sampling:
        hatW_sample_size = len(hatW_samples)
    else:
        hatW_sample_size = redo_sampling_size
        hatW_samples=generate_hatW_samples(M, redo_sampling_size, seeding=False)
    print('ZZZ: ', hatW_sample_size)
    
    # Mtrx_Norm = np.max(np.abs(Mtrx))
    # Mtrx_rn = Mtrx/Mtrx_Norm
    # # Mtrx_rn_inv = np.linalg.inv(Mtrx_rn)
    # Mtrx_rn_inv = scplin.inv(Mtrx_rn)
    # Mtrx_inv = Mtrx_rn_inv/Mtrx_Norm

    # Mtrx_inv_bis = scplin.inv(Mtrx)
    Mtrx_inv = scplin.inv(Mtrx)

    plt.figure()
    plt.title('inLasso Mtrx')
    plt.imshow(Mtrx)
    plt.colorbar()

    plt.figure()
    plt.title('inLasso Mtrx_inv')
    plt.imshow(Mtrx_inv)
    plt.colorbar()

    plt.figure()
    plt.title('inLasso I1')
    # plt.imshow(np.dot(Mtrx_rn, Mtrx_rn_inv))
    plt.imshow(np.dot(Mtrx, Mtrx_inv))
    plt.colorbar()

    plt.figure()
    plt.title('inLasso I2')
    # plt.imshow(np.dot(Mtrx_rn_inv, Mtrx_rn))
    plt.imshow(np.dot(Mtrx_inv, Mtrx))
    plt.colorbar()

    w_shift = np.dot(Mtrx_inv, B_vct)
    Coeff_C = np.sum(Y*Y) - np.dot(B_vct, w_shift)

    print('w_shif: ', w_shift)

    hatW_shifted = [ item/(beta*tilde_lambda) - w_shift for item in hatW_samples ]
    print('XXX', len(hatW_shifted))

    nll1 = 0.5*beta*Coeff_C/float(N) - 0.5*(1.0-1.0/float(N))*log(beta)
    nll2 = 0.5*log(2*np.pi)*(N-1)/N + 0.5*log(N)/N
    
    coef_hatZ = 0.0
    ave_ws = np.zeros(M)

    exponts = np.array([-0.5*beta*np.dot(item, np.dot(Mtrx, item)) for item in hatW_shifted])
    print('max / min expo: ', np.max(exponts), ' / ', np.min(exponts))
    exponts.sort()
    # plt.figure()
    # plt.title('exponents')
    # plt.hist(exponts, bins=20, density=1, alpha=0.7)
    # plt.plot(-exponts, np.linspace(0,1,len(exponts)), 'x')
    log_coef = log(1e4) - np.max(exponts) 
    mutted_exponts = exponts + log_coef
    for i in range(len(hatW_shifted)):
        item = hatW_shifted[i]
        # print('item: ', item)
        coef_dZ = exp(mutted_exponts[i])
        print('dZ: ', coef_dZ, 'mutted exponent: ', mutted_exponts[i], ' exponts.', exponts[i])
        coef_hatZ = coef_hatZ + coef_dZ
        ave_ws = ave_ws + coef_dZ*(item+w_shift)
    coef_hatZ = coef_hatZ / hatW_sample_size
    print('coef hat Z ', coef_hatZ)
    ave_ws = ave_ws / hatW_sample_size / coef_hatZ
    nll3 = -(log(coef_hatZ)-log_coef)/N

    print('nll1: {:.4E}, nll2: {:.4E}, nll3: {:.4E}'.format(nll1, nll2, nll3) )

    nll = nll1 + nll2 + nll3

    return {'nll':nll, 'Ave_Ws': ave_ws}
#g
def Bayes_LR_Lasso_NLL_MG(ElementaryQuantities, tilde_lambda, beta, W_samples:list, redo_sampling:bool=False, redo_sampling_size:int=1000):
    N = ElementaryQuantities['N']
    M = ElementaryQuantities['M']
    Mtrx = ElementaryQuantities['Mtrx']
    B_vct = ElementaryQuantities['B']
    ave_target = ElementaryQuantities['ave_y']
    ave_feature = ElementaryQuantities['ave_feature']
    Y = ElementaryQuantities['Y']

    log_det_Mtrx = ElementaryQuantities['log_det_M']
    Mtrx_inv = ElementaryQuantities['Mtrx_inv']
    w_shift = ElementaryQuantities['mean_ws']
    # w_shift = np.dot(Mtrx_inv, B_vct)

    mc_size = len(W_samples)

    # log_Mtrx_norm = 0 
    # for i in range(M):
    #     log_Mtrx_norm = log_Mtrx_norm + log(abs(Mtrx[i,i]))
    # av_log_Mtrx_norm = log_Mtrx_norm/M 
    # Mnorm = exp(av_log_Mtrx_norm)
    # Mtrx_rn = Mtrx/Mnorm 
    # det_Mtrx_rn = np.linalg.det(Mtrx_rn)
    # log_det_Mtrx = M*log(Mnorm) + log(det_Mtrx_rn)

    # # Mtrx_inv_bis = scplin.inv(Mtrx)
    # Mtrx_inv = scplin.inv(Mtrx)

    # plt.figure()
    # plt.title('inLasso_MG Mtrx')
    # plt.imshow(Mtrx)
    # plt.colorbar()

    # plt.figure()
    # plt.title('inLasso_MG Mtrx_inv')
    # plt.imshow(Mtrx_inv)
    # plt.colorbar()

    # plt.figure()
    # plt.title('inLasso_MG I1')
    # # plt.imshow(np.dot(Mtrx_rn, Mtrx_rn_inv))
    # plt.imshow(np.dot(Mtrx, Mtrx_inv))
    # plt.colorbar()

    # plt.figure()
    # plt.title('inLasso_MG I2')
    # # plt.imshow(np.dot(Mtrx_rn_inv, Mtrx_rn))
    # plt.imshow(np.dot(Mtrx_inv, Mtrx))
    # plt.colorbar()

    Coeff_C = np.sum(Y*Y) - np.dot(B_vct, w_shift)
    # print('w_shif: ', w_shift)

    nll1 = 0.5*beta*Coeff_C/float(N) - 0.5*(1.0-1.0/float(N))*log(beta)
    nll2 = 0.5*log(2*np.pi)*(N-1)/N + 0.5*log(N)/N
    
    Intgrl_N = 0.0
    ave_ws = np.zeros(M)

    d_Intgrl_coeff_list = []
    d_Intgrl_expnt_list = []
    d_ave_ws_coeff_list = []
    for i in range(mc_size): 
        w_vct = W_samples[i]
        theX = -beta*tilde_lambda*np.sum(np.abs(w_vct))
        d_Intgrl_sci = convertor_base10(theX)
        d_Intgrl_coeff_list.append(d_Intgrl_sci['coefficient'])
        d_Intgrl_expnt_list.append(d_Intgrl_sci['exponent'])
        d_ave_ws_coeff_list.append(w_vct*d_Intgrl_sci['coefficient'])
        
        d_Intgrl = exp(theX)
        # print('dIntg ', d_Intgrl)
        print('exp(-{:.2E})'.format((beta*tilde_lambda)**0*np.sum(np.abs(w_vct))))
        Intgrl_N = Intgrl_N + d_Intgrl
        ave_ws = ave_ws + d_Intgrl*w_vct

    print(np.unique(np.array(d_Intgrl_expnt_list), return_counts=1))
    
    Intgrl_N = Intgrl_N/mc_size
    print(Intgrl_N)
    ave_ws = ave_ws/mc_size/Intgrl_N
    
    if Intgrl_N==0: ## debugging
        print('Intgrl_N=0')
        sys.exit() 
    
    nll3 = (-1.0/float(N))*( 0.5*M*log(beta) + M*log(tilde_lambda) + 0.5*M*log(np.pi*0.5) - 0.5*log_det_Mtrx + log(Intgrl_N) )

    # print('nll1: {:.4E}, nll2: {:.4E}, nll3: {:.4E}'.format(nll1, nll2, nll3) )

    nll = nll1 + nll2 + nll3

    return {'nll':nll, 'Ave_Ws': ave_ws}

# def Bayes_LR_Ridge_ExtraFeature_ElemtaryQuantities(targets, features, print_info: bool=False, plot_info: bool=True):
#     N = shape(targets)[0]
#     N_f = shape(features)[0]
#     M_raw = shape(features)[1]
#     M = M_raw+1
#     features_hat = np.zeros([N,M])
#     features_hat[:,0] = 1.0
#     features_hat[:,1:] = features

#     if N!=N_f :
#         print('N_targets != N_features, number of data points error!')
#         sys.exit()

#     ave_target = np.mean(targets)
#     ave_square_target = np.mean(targets**2)
#     b0 = np.zeros(M)
#     b1 = np.zeros(M)
#     for i in range(M):
#         b0[i] = np.sum(targets*features_hat[:,i])
#         b1[i] = -np.sum(features_hat[:,i])
#     Mtrx = np.zeros([M,M])
#     for i in range(M):
#         for j in range(M):
#             Mtrx[i,j] = np.sum(features_hat[:,i]*features_hat[:,j])

#     return {'Mtrx': Mtrx, 'b0': b0, 'b1':b1, 'ave_y': ave_target, 'ave_y2': ave_square_target, 'N': N, 'M': M}

# def K_related(ElementaryQuantities, tilde_lambda, model_option:str='w0=hyper', print_info: bool=False, plot_info: bool=False):
#     N = ElementaryQuantities['N']
#     M = ElementaryQuantities['M']
#     Mtrx = ElementaryQuantities['Mtrx']
#     b0 = ElementaryQuantities['b0']
#     b1 = ElementaryQuantities['b1']
#     ave_target = ElementaryQuantities['ave_y']
#     ave_square_target = ElementaryQuantities['ave_y2']

#     K_Mtrx = Mtrx + tilde_lambda*np.eye(M)
#     K_Mtrx_inv = np.linalg.inv(K_Mtrx)

#     # print('det 1 : ')
#     det_K = np.linalg.det(K_Mtrx)

#     log_Knorm = 0
#     for i in range(M):
#         log_Knorm = log_Knorm + log(abs(K_Mtrx[i,i]))
#     av_log_Knorm = log_Knorm / M
#     Knorm = exp(av_log_Knorm)
#     K_rn = K_Mtrx/Knorm

#     # print('det 2 : ')
#     det_K_rn = np.linalg.det(K_rn)
#     log_det_K = M*log(Knorm) + log(det_K_rn)

#     Coeff_A = N - np.dot(b1, np.dot(K_Mtrx_inv, b1))
#     Coeff_B = N*ave_target + np.dot(b0, np.dot(K_Mtrx_inv, b1))
#     Coeff_B_bis = N*ave_target + np.dot(b1, np.dot(K_Mtrx_inv, b0))
#     Coeff_C = N*ave_square_target - np.dot(b0, np.dot(K_Mtrx_inv, b0))

#     w0_opt = Coeff_B/Coeff_A
#     w0_opt_bis = Coeff_B_bis/Coeff_A
#     beta_opt = float(N)/(Coeff_C - Coeff_B**2/Coeff_A)
#     beta_opt_bis = float(N)/(Coeff_C - Coeff_B_bis**2/Coeff_A)

#     ell_1 = -M*log(tilde_lambda)/float(2*N) 
#     ell_2 = log_det_K/float(2*N) 
#     ell_2_bis = log(det_K)/float(2*N)
#     ell_3 = 0.5*log((Coeff_C-Coeff_B**2/Coeff_A)/float(N))
#     ell_4 = 0.5+0.5*log(2*np.pi)

#     nll = ell_1+ell_2+ell_3+ell_4
#     nll_bis = ell_1+ell_2_bis+ell_3+ell_4

#     off_set_key = ''
#     if model_option=='w0=hyper':
#         off_set_key = 'w0'
#     elif model_option=='extra_feature':
#         off_set_key = 'mu'
#     else:
#         print('model_option: {} not existing !'.format(model_option))
#         sys.exit()
#     return {'K': K_Mtrx, 'K_inv': K_Mtrx_inv, 'A': Coeff_A, 'B': Coeff_B, 'B_bis': Coeff_B_bis, 'C': Coeff_C, 'det_K': det_K, 'det_K_rn': det_K_rn, 'log_det_K': log_det_K, 'l1':ell_1, 'l2':ell_2, 'l2_bis': ell_2_bis, 'l3': ell_3, 'Knorm':Knorm, 'std_Krn': np.std(K_rn), 'nll': nll, 'nll_bis': nll_bis, 'beta': beta_opt, 'beta_bis':beta_opt_bis, off_set_key: w0_opt, off_set_key+'_bis': w0_opt_bis, 'off_set_key': off_set_key}

# def K_related_DoubleLambda(ElementaryQuantities, tilde_lambda, tilde_lambda_0, print_info: bool=False, plot_info: bool=False):
#     N = ElementaryQuantities['N']
#     M = ElementaryQuantities['M']
#     Mtrx = ElementaryQuantities['Mtrx']
#     b0 = ElementaryQuantities['b0']
#     b1 = ElementaryQuantities['b1']
#     ave_target = ElementaryQuantities['ave_y']
#     ave_square_target = ElementaryQuantities['ave_y2']

#     I_DoubleLambda = tilde_lambda*np.eye(M)
#     I_DoubleLambda[0,0] = tilde_lambda_0
#     K_Mtrx = Mtrx + I_DoubleLambda
#     K_Mtrx_inv = np.linalg.inv(K_Mtrx)

#     # print('det 1 : ')
#     det_K = np.linalg.det(K_Mtrx)

#     log_Knorm = 0
#     for i in range(M):
#         log_Knorm = log_Knorm + log(abs(K_Mtrx[i,i]))
#     av_log_Knorm = log_Knorm / M
#     Knorm = exp(av_log_Knorm)
#     K_rn = K_Mtrx/Knorm
    
#     # print('det 2 : ')
#     det_K_rn = np.linalg.det(K_rn)
#     log_det_K = M*log(Knorm) + log(det_K_rn)

#     # Coeff_A = N - np.dot(b1, np.dot(K_Mtrx_inv, b1))
#     # Coeff_B = N*ave_target + np.dot(b0, np.dot(K_Mtrx_inv, b1))
#     # Coeff_B_bis = N*ave_target + np.dot(b1, np.dot(K_Mtrx_inv, b0))
#     Coeff_C = N*ave_square_target - np.dot(b0, np.dot(K_Mtrx_inv, b0))

#     # w0_opt = Coeff_B/Coeff_A
#     # w0_opt_bis = Coeff_B_bis/Coeff_A
#     beta_opt = float(N)/Coeff_C 
#     # beta_opt_bis = float(N)/(Coeff_C - Coeff_B_bis**2/Coeff_A)

#     ell_1 = -M*log(tilde_lambda)/float(2*N)
#     ell_2 = log_det_K/float(2*N) 
#     ell_2_bis = log(det_K)/float(2*N)
#     ell_3 = 0.5*log(Coeff_C/float(N))
#     ell_4 = 0.5+0.5*log(2*np.pi)

#     nll = ell_1+ell_2+ell_3+ell_4
#     nll_bis = ell_1+ell_2_bis+ell_3+ell_4

#     return {'K': K_Mtrx, 'K_inv': K_Mtrx_inv, 'C': Coeff_C, 'det_K': det_K, 'det_K_rn': det_K_rn, 'log_det_K': log_det_K, 'l1':ell_1, 'l2':ell_2, 'l2_bis': ell_2_bis, 'l3': ell_3, 'Knorm':Knorm, 'std_Krn': np.std(K_rn), 'nll': nll, 'nll_bis': nll_bis, 'beta': beta_opt}

# def Posterior_MeanCov_Ridge(R, E): 
#     # R = return of Bayes_LR_Ridge_betaW0Hyper_betaW0Optimal, E=return of Bayes_LR_Ridge_betaW0Hyper_ElemtaryQuantities
#     off_set_key = R['off_set_key']
#     Mean = np.dot(R['K_inv'], E['b0']+R[off_set_key]*E['b1'])
#     Cov_Mtrx = R['K_inv']/R['beta']

#     return {'mean': Mean, 'cov_mtrx': Cov_Mtrx}
#g
def Linear_Regression_Ridge_HomeMade(targets, features, panelty_coeff_ratio, stop_crt, max_iter_in_N=10, dt=0.1, init_coeffs=-1, init_offset=0, rand_seed=1273291610, print_info: bool=False, plot_info: bool=True, method: str='matrix inverse'):
    N = shape(targets)[0]
    N_f = shape(features)[0]
    M = shape(features)[1]

    if N!=N_f :
        print('N_targets != N_features, number of data points error!')
        sys.exit()
    
    # if init_coeffs==-1 :
    #     init_coeffs = np.ones(M)
    # elif len(init_coeffs)!= M :
    #     print('len(init_weights)!= M+1, number of weights error!')
    #     sys.exit()
    if len(init_coeffs)!= M :
        print('len(init_weights)!= M+1, number of weights error!')
        sys.exit()

    # panelty_coeff = $\tilde{\lambda}$
    panelty_coeff = panelty_coeff_ratio

    Ws = copy(init_coeffs)
    W0 = init_offset

    if print_info:
        print("w0 = ", W0)
        print("Ws = ", Ws)

    ave_target = np.mean(targets)
    ave_feature = np.zeros(M)
    for k in range(M):
        ave_feature[k] = np.mean(features[:,k])
    centered_features = copy(features)
    for i in range(N):
        centered_features[i,:] = features[i,:] - ave_feature
    centered_targets = targets - ave_target
    Cnst = copy(Ws)
    for i in range(M):
        Cnst[i] = np.sum(centered_targets*centered_features[:,i])
    Mtrx = np.zeros([M,M])
    for i in range(M):
        for j in range(M):
            Mtrx[i,j] = np.sum(centered_features[:,i]*centered_features[:,j])

    final_Ws = np.zeros(M)
    final_W0 = 0.0
    final_gradWs = np.ones(M)

    iter_step = 0
    Max_Steps = int(max_iter_in_N*N)

    if method == 'matrix inverse':
        # M_inv = np.linalg.inv(Mtrx+np.eye(M)*panelty_coeff)
        # M_inv = scplin.inv(Mtrx+np.eye(M)*panelty_coeff, check_finite=True)
        # final_Ws_by_inverse = np.dot(M_inv, Cnst)
        final_Ws = scplin.solve(Mtrx+np.eye(M)*panelty_coeff, Cnst) # np.dot(M_inv, Cnst)  # 
        final_W0 = ave_target - np.dot(final_Ws ,ave_feature)
        final_gradWs = Cnst - np.dot(Mtrx, final_Ws) - panelty_coeff*final_Ws
        wghts_acc = np.mean(np.abs(final_gradWs))
        if False : #plot_info:
            print(np.dot(M_inv, Mtrx+np.eye(M)*panelty_coeff))
            plt.figure()
            plt.title('M_inv * Mtrx+1*panelty')
            plt.imshow(np.dot(M_inv, Mtrx+np.eye(M)*panelty_coeff))
            plt.colorbar()
            plt.figure()
            plt.title('Mtrx+1*panelty')
            plt.imshow(Mtrx+np.eye(M)*panelty_coeff)
            plt.colorbar()
            plt.figure()
            plt.title('inverse of Mtrx+1*panelty')
            plt.imshow(M_inv)
            plt.colorbar()
            plt.figure()
            plt.title('compare Ws by solve and inv')
            plt.plot(final_Ws,'x', ms=8)
            plt.plot(final_Ws_by_inverse, '+', ms=8)

    elif method == 'global gradient descent':

        Max_Steps = int(max_iter_in_N*N)
        iter_step = 0
        wghts_acc = 1000
        if print_info: print('global gradient descent: to start while loop, start? ', iter_step<Max_Steps and wghts_acc>stop_crt)
        # np.random.seed(rand_seed)
        while iter_step<Max_Steps and wghts_acc>stop_crt :
            
            gradWs = ( Cnst - np.dot(Mtrx, Ws) - panelty_coeff*Ws)
            delta_w =  dt * gradWs
            new_Ws = Ws + delta_w

            iter_step += 1
    
            # wghts_acc = Norm_HD(delta_w)/Norm_HD(Ws)
            # wghts_acc = np.max(np.abs(delta_w/Ws))
            # wghts_acc = max(np.abs(delta_w0), np.max(np.abs(delta_w)))/dt
            wghts_acc = np.max(np.abs(gradWs))

            if iter_step%(N*100)==0 and print_info:
                print('iter_step = ', iter_step, ' w_err = ', wghts_acc)
                # print('old: ', Norm_HD(old_w), ' , ', np.mean(np.abs(old_w)))
                # print('delta: ', Norm_HD(delta_w), ' , ', np.mean(np.abs(delta_w)))
            
            Ws = copy(new_Ws)
        if print_info: print('while loop ends.')
        final_Ws = new_Ws # - 0.5*delta_w
        final_W0 = ave_target - np.dot(Ws,ave_feature)

    elif method == 'partial gradient descent':

        Max_Steps = int(max_iter_in_N*M)
        iter_step = 0
        wghts_acc = 1000
        if print_info: print('partial gradient descent: to start while loop, start? : ', iter_step<Max_Steps and wghts_acc>stop_crt)
        # np.random.seed(rand_seed)
        gradWs = ( Cnst - np.dot(Mtrx, Ws) - panelty_coeff*Ws)
        while iter_step<Max_Steps and wghts_acc>stop_crt :
            
            idx = iter_step%M
            Ws[idx] = (Cnst[idx] - np.dot(Mtrx[idx,:], Ws) + Mtrx[idx,idx]*Ws[idx])/(Mtrx[idx,idx] + panelty_coeff)
            if idx==0 or iter_step==Max_Steps-1:
                gradWs = ( Cnst - np.dot(Mtrx, Ws) - panelty_coeff*Ws)
                wghts_acc = np.mean(np.abs(gradWs))

            iter_step += 1

            if iter_step%(M*100)==0 and print_info:
                print('iter_step = ', iter_step, ' w_err = ', wghts_acc)
                # print('old: ', Norm_HD(old_w), ' , ', np.mean(np.abs(old_w)))
                # print('delta: ', Norm_HD(delta_w), ' , ', np.mean(np.abs(delta_w)))
            
        if print_info: print('while loop ends.')
        final_Ws = Ws 
        final_W0 = ave_target - np.dot(Ws,ave_feature)

    else:
        print('paramter [ method ] takes a wrong value ...')
        sys.exit()

    emp_risk = 0
    for i in range(N):
        emp_risk = emp_risk + (targets[i]  - final_W0 - np.sum(final_Ws*features[i,:]) )**2 / N

    return {'weights': final_Ws, 'offset': final_W0, 'accuracy': wghts_acc, 'reach_wall': iter_step>=Max_Steps, 'emp_risk': emp_risk, 'grad_weights': final_gradWs, 'b': Cnst, 'M': Mtrx, 'expression of grad_w': 'b - M*Ws - panelty_coeff*Ws'}

# def Linear_Regression_Lasso_HomeMade(targets, features, panelty_coeff_ratio, stop_crt, max_iter_in_N=10, dt=0.1, init_coeffs=-1, init_offset=0, rand_seed=1273291610, print_info: bool=False, plot_info: bool=True, method:str=''):
#     N = shape(targets)[0]
#     N_f = shape(features)[0]
#     M = shape(features)[1]

#     if N!=N_f :
#         print('N_targets != N_features, number of data points error!')
#         sys.exit()
    
#     # if init_coeffs==-1 :
#     #     init_coeffs = np.ones(M)
#     # elif len(init_coeffs)!= M :
#     #     print('len(init_weights)!= M+1, number of weights error!')
#     #     sys.exit()
#     if len(init_coeffs)!= M :
#         print('len(init_weights)!= M+1, number of weights error!')
#         sys.exit()

#     panelty_coeff = panelty_coeff_ratio

#     Ws = copy(init_coeffs)
#     W0 = init_offset

#     if print_info:
#         print("w0 = ", W0)
#         print("Ws = ", Ws)

#     ave_target = np.mean(targets)
#     ave_feature = np.zeros(M)
#     for k in range(M):
#         ave_feature[k] = np.mean(features[:,k])
#     centered_features = copy(features)
#     for i in range(N):
#         centered_features[i,:] = features[i,:] - ave_feature
#     centered_targets = targets - ave_target
#     Cnst = copy(Ws)
#     for i in range(M):
#         Cnst[i] = np.sum(centered_targets*centered_features[:,i])
#     Mtrx = np.zeros([M,M])
#     for i in range(M):
#         for j in range(M):
#             Mtrx[i,j] = np.sum(centered_features[:,i]*centered_features[:,j])

#     final_Ws = np.zeros(M)
#     final_W0 = 0.0
#     final_gradWs = np.ones(M)

#     iter_step = 0
#     Max_Steps = int(max_iter_in_N*N)

#     if method == 'global gradient descent':

#         Max_Steps = int(max_iter_in_N*N)
#         iter_step = 0
#         wghts_acc = 1000
#         if print_info: print('global gradient descent: to start while loop, start? ', iter_step<Max_Steps and wghts_acc>stop_crt)
#         # np.random.seed(rand_seed)
#         while iter_step<Max_Steps and wghts_acc>stop_crt :
            
#             sign_w = np.sign(Ws)
#             gradWs = ( Cnst - np.dot(Mtrx, Ws) - panelty_coeff*sign_w)
#             delta_w =  (dt/N) * gradWs
#             new_Ws = Ws + delta_w

#             iter_step += 1
    
#             # wghts_acc = Norm_HD(delta_w)/Norm_HD(Ws)
#             # wghts_acc = np.max(np.abs(delta_w/Ws))
#             # wghts_acc = max(np.abs(delta_w0), np.max(np.abs(delta_w)))/dt
#             wghts_acc = np.max(np.abs(delta_w))

#             # if print_info: print('iter:', iter_step, 'mean_abs_grad = ', np.mean(np.abs(gradWs)))

#             if iter_step%(N*100)==0 and print_info:
#                 print('iter_step = ', iter_step, ' w_err = ', wghts_acc)
#                 # print('old: ', Norm_HD(old_w), ' , ', np.mean(np.abs(old_w)))
#                 # print('delta: ', Norm_HD(delta_w), ' , ', np.mean(np.abs(delta_w)))
            
#             Ws = copy(new_Ws)
#         if print_info: print('while loop ends.')
#         final_Ws = new_Ws # - 0.5*delta_w
#         final_W0 = ave_target - np.dot(Ws,ave_feature)

#     elif method == 'partial gradient descent':
#         pass
#     else:
#         print('paramter [ method ] takes a wrong value ...')
#         sys.exit()

#     if print_info: print('emp risk')
#     emp_risk = 0
#     for i in range(N):
#         # if print_info: print(i, end=' ')
#         emp_risk = emp_risk + (targets[i]  - final_W0 - np.sum(final_Ws*features[i,:]) )**2 / N

#     return {'weights': final_Ws, 'offset': final_W0, 'accuracy': wghts_acc, 'reach_wall': iter_step>=Max_Steps, 'emp_risk': emp_risk, 'grad_weights': final_gradWs, 'b': Cnst, 'M': Mtrx, 'expression of grad_w': 'b - M*Ws - panelty_coeff*Ws'}

# def Simple_Linear_Regressor(targets, features, panelty_coeff, stop_crt, max_iter_in_N=10, dt=0.1, init_weights=-1, rand_seed=378274675):
#     N = shape(targets)[0]
#     N_f = shape(features)[0]
#     M = shape(features)[1]

#     print('M=', M)

#     if N!=N_f :
#         print('N_t != N_f, number of data points error!')
#         sys.exit()
    
#     if init_weights==-1 :
#         init_weights = np.ones(M+1)
#     elif len(init_weights)!= M+1 :
#         print('len(init_weights)!= M+1, number of weights error!')
#         sys.exit()
    
#     old_w = copy(init_weights)
#     new_w = copy(init_weights)
#     adp_features = np.zeros([N,M+1])
#     adp_features[:,1:] = features
#     adp_features[:,0] = np.ones(N)

#     print('shape of adp_features: ', shape(adp_features))

#     Max_Steps = int(max_iter_in_N*N)

#     iter_step = 0
#     wghts_acc = 1000
#     print('to start while loop, start? : ', iter_step<Max_Steps and wghts_acc>stop_crt)
#     np.random.seed(rand_seed)
#     while iter_step<Max_Steps and wghts_acc>stop_crt :
        
#         # sequential update
#         idx = int(np.random.uniform(0,N))  # iter_step%N
#         regulator = panelty_coeff*old_w
#         regulator[0] = 0.0
#         delta_w = dt* ((targets[idx] - np.sum(old_w*adp_features[idx,:]))*adp_features[idx,:] - regulator)
#         new_w = old_w + delta_w

#         # parallel update
#         # delta_w = -dt*panelty_coeff*old_w 
#         # for idx in range(N):
#         #     delta_w = delta_w + dt*(targets[idx] - np.sum(old_w*adp_features[idx,:]))*adp_features[idx,:]
#         # new_w = old_w + delta_w

#         iter_step += 1
#         # wghts_acc = Norm_HD(delta_w)/Norm_HD(old_w)
#         wghts_acc = np.max(np.abs(delta_w/old_w))

#         if iter_step%5000==0:
#             print('iter_step = ', iter_step, ' w_err = ', wghts_acc)
#             # print('old: ', Norm_HD(old_w), ' , ', np.mean(np.abs(old_w)))
#             # print('delta: ', Norm_HD(delta_w), ' , ', np.mean(np.abs(delta_w)))
        
#         old_w = copy(new_w)
#     print('while loop ends.')
#     final_w = new_w - 0.5*delta_w

#     emp_risk = 0
#     for i in range(N):
#         emp_risk = emp_risk + (targets[i]  - np.sum(final_w*adp_features[i,:]) )**2 / N

#     return {'weights': final_w, 'accuracy': wghts_acc, 'reach_wall': iter_step>=Max_Steps, 'emp_risk': emp_risk, 'adp_features': adp_features}

# def linear_target_function(weights, adp_features):
#     return np.sum(weights*adp_features)