import numpy as np
import scipy as scp
import math as ma
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys
import os.path
import imp
import json
from scipy.optimize import curve_fit

import sklearn.cluster as skcltr
import sklearn.linear_model as skl_linear

import torch
import torch.nn as nn
import torch.nn.functional as nF
import torchvision
import torchvision.transforms as transforms

# for data loading
import data_read as dr 

# for cross-validation data preparation 
import data_prep_crossvalid as dxv

# handmade linear regressors
import simpleLinearReg as slr

# import functions from other training methods
import GaussianLH_Panelty_RidgeLasso_MAP as glh_map

imp.reload(dr)
imp.reload(dxv)
imp.reload(slr)
imp.reload(glh_map)

def Sine_Data_Generator(N, M, noise=1.0, seeding=102939028, x_low_bound=-np.pi, x_high_bound=np.pi):
    
    gf = lambda x: np.sin(x)

    np.random.seed(seeding)
    x_data = np.random.uniform(x_low_bound, x_high_bound, N)
    add_noise = np.random.normal(0, 0.2, N)
    y_data = gf(x_data) + add_noise*noise
    features = np.array([[itm**(k+1) for k in range(M)] for itm in x_data])
    targets = np.array([[item] for item in y_data])

    plt.figure()
    plt.title('training data')
    plt.plot(x_data, y_data, 'o', mfc='none', mew=1.5)

    return features, targets

def Polynomial_Data_Generator(N, M, m, coefs, cst, x_low_bound=-np.pi, x_high_bound=np.pi, noise=1.0, seeding=102939028):
    if m>len(coefs):
        print('input error type I - Polynomial_Data_Generator - ')
    if M>m :
        print('input error type II - Polynomial_Data_Generator - ')
    
    def f_poly(x):
        r = cst*np.ones(len(x))
        for i in range(m):
            r = r + coefs[i]*x**(i+1)
        return r

    np.random.seed(seeding)
    add_noise = np.random.normal(0, 0.2, N)
    x_data = np.random.uniform(x_low_bound, x_high_bound, N)
    y_data = f_poly(x_data) + add_noise*noise
    features = np.array([[ item**(k+1) for k in range(M) ] for item in x_data])
    targets = np.array([[item] for item in y_data])

    plt.figure()
    plt.title('training data')
    plt.plot(x_data, y_data, 'o', mfc='none', mew=1.5)

    return features, targets


def RandomSplit_CrossVal_Preparation(N, features, targets, set_num_trials=1, set_valid_ratio=0.1):

    cross_validation_method = 'rand_cross'
    parameter_crossvalidation = {
        'validation_ratio' : 0.1, #set_valid_ratio, # its function depends on the opted cross-validation method. 

        ## for Nest Cross Validation ##
        'num_cell' : int(5), # number of cells in the "nest"

        ## for Segmenting Cross Validation ##
        'num_seg' : int(3), # number of segments apart from validation set

        ## for Random Split Cross Validation ##
        'num_split_ratio': set_num_trials*set_valid_ratio,  # to generate number of trials : num_split_ratio/validation_ratio

        ## Producibility of randomness ##
        'rand_seed': 357828922 , 
        'seeding_or_not': True,   # true for reproducible cross validation data structure using 'rand_seed'

        ## Control Options ##
        'print_info_or_not': True
    }

    ### Return Data for Cross Validation ###
    return glh_map.DataPreparation_for_CrossValidation(cross_validation_method, N, features, targets, parameter_crossvalidation)

class LinaerModel_Mto1():
    def __init__(self, M):
        self.num_ftrs = M
        self.Ws = torch.zeros(M, requires_grad=True, dtype=torch.float64)
        self.W0 = torch.zeros(1, requires_grad=True, dtype=torch.float64)
    
    def forward(self, X:torch.Tensor): 
        X.double()
        return torch.matmul(X, self.Ws) + self.W0

    def zero_grad(self):
        if self.Ws.grad !=None : self.Ws.grad.zero_()
        if self.W0.grad !=None : self.W0.grad.zero_()
    
    def Predict(self, X:torch.Tensor):
        X.double()
        with torch.no_grad():
            return torch.matmul(X, self.Ws) + self.W0

    def Update(self, lr):
        # lrs = lr/self.Ws.grad.abs().max()
        with torch.no_grad():
            self.Ws -= lr * self.Ws.grad
            self.W0 -= lr * self.W0.grad

def MeanSqureLoss(T, y):
    return ((T - y)**2).mean()

def MeanAbsLoss(T, y):
    return (T-y).abs().mean()

def Ridge_Reg(penalty, lm:LinaerModel_Mto1):
    return  penalty * (lm.Ws**2).sum()

def Train_LinearModel_Mto1(lm:LinaerModel_Mto1, M, pnlt, ftr, trg, lr, Max_steps=1000, num_print = 10, crt_stop=1e-5):
    itr = 0
    err = 100000.
    print_step = int(Max_steps/num_print) if int(Max_steps/num_print)>0 else 1
    while itr<Max_steps and err>crt_stop:
        y_frwrd = lm.forward(ftr)
        loss = MeanSqureLoss(trg, y_frwrd) + Ridge_Reg(pnlt, lm)
        loss.backward()
        lm.Update(lr)

        err = lm.Ws.grad.abs().mean().item()*M/float(M+1)+lm.W0.grad.abs().mean().item()/float(M+1)
        
        if itr%print_step==0:
            # print(itr) 
            print('itr.{:d} '.format(itr), 'err{:.3f} , loss.{:.3f}'.format(err, loss.item()))
        # print(lm.Ws.grad)
        
        lm.zero_grad()

        itr = itr+1
    else:
        print('Training loop finished: itr.{:d} / Max_Steps.{:d}, err.{:.3f} Vs crt_stop.{:.3f}'.format(itr, Max_steps, err, crt_stop))

    return lm, err

def Valid_LinearModel_Mto1(lm:LinaerModel_Mto1, ftr, trg):
    with torch.no_grad():
        y_prd = lm.Predict(ftr)
        l = MeanSqureLoss(trg, y_prd)    
    return l

class LinearRegressionFromNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearRegressionFromNN, self).__init__() 
        self.lin = nn.Linear(in_dim, out_dim, dtype=torch.float64)

    def forward(self, X):
        return self.lin(X)

def main_handcrafted_LinearRegression_usingClass():
    pass

    N=200
    M=5
    m=5
    the_coefs=[2.0, 1.5 ,0,0,0]
    the_bias = 1.0
    # features, tarts = Polynomial_Data_Generator(N, M, m, the_coefs, the_bias)
    features, tarts = Sine_Data_Generator(N,M,0.1)
    targets = np.array([item[0] for item in tarts]) 

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
    num_trial, Data_Cluster = glh_map.DataPreparation_for_CrossValidation(cross_validation_method, N, features, targets, parameter_crossvalidation)

    ### Training ###
    ### Ridge MAP - PyTorch ###
    ftr_train_tnsr = torch.tensor(Data_Cluster[0]['fit']['feature'])
    trg_train_tnsr = torch.tensor(Data_Cluster[0]['fit']['target'])
    ftr_valid_tnsr = torch.tensor(Data_Cluster[0]['valid']['feature'])
    trg_valid_tnsr = torch.tensor(Data_Cluster[0]['valid']['target'])

    lr = 1e-4
    
    lm = LinaerModel_Mto1(M)
    lm2 = LinaerModel_Mto1(M)

    ### Training lm -- approch: manual ##
    fig_y, ax_y = plt.subplots()
    fig_ws, ax_ws = plt.subplots()
    ax_y.set_title('y_predict vs train targets')
    ax_ws.set_title('grad ws')
    # ax_y.set_yscale('log')
    # ax_ws.set_yscale('log')
    
    lm.zero_grad()
    tot_steps = 100000
    plot_step = 5000
    clrs=cm.rainbow(np.linspace(0,1,tot_steps))
    for k in range(tot_steps):
        
        y_fr = lm.forward(ftr_train_tnsr)
        if k%plot_step==0 or k==tot_steps-1:
            ax_y.plot(trg_train_tnsr.detach(), y_fr.detach(), 'o', ms=7, color=clrs[k], mfc='none', alpha=0.6)

        # loss = ((trg_train_tnsr - y_fr)**2).mean()
        loss = MeanSqureLoss(trg_train_tnsr, y_fr)
        
        if k%plot_step==0 or k==tot_steps-1: 
            print('loss=', loss.item())
            if lm.Ws.grad!=None:
                ax_ws.plot(lm.Ws.grad.detach(), 'x', ms=6, color=clrs[k],)
            else:
                ax_ws.plot(np.zeros(M), 'x', ms=6, color=clrs[k],)
        
        loss.backward()
        
        if k%plot_step==0 or k==tot_steps-1: 
            print('after loss.backward(): grad_ws: ', lm.Ws.grad.abs().mean().item())
            ax_ws.plot(lm.Ws.grad.detach(), 'o', ms=6, color=clrs[k], mfc='none')
        
        if k%plot_step==0: print('before one update: grad_ws: ', lm.Ws.mean().item())
        
        lm.Update(lr=lr)
        
        if k%plot_step==0: print('after one update: grad_ws: ', lm.Ws.mean().item())
        
        lm.zero_grad()
        # print('after zero out grad: grad_ws: ', lm.Ws.grad.abs().mean().item())

    y_test = lm.Predict(ftr_valid_tnsr)
    fig, ax = plt.subplots()
    ax.set_title('y vs test target')
    ax.plot(trg_valid_tnsr.detach(), y_test.detach(), 'ro', ms=10, mfc='none', mew=2)
    ax.grid(True, which='both')


    ## Training lm2 -- approach: using the class ##
    y2tst = lm2.Predict(ftr_train_tnsr)
    fig_y2, ax_y2 = plt.subplots()
    ax_y2.set_title('lm2 y vs targets - training data')
    ax_y2.plot(trg_train_tnsr.detach(), y2tst.detach(), 'ro', ms=7, mfc='none', label='pre-training')
    
    print('train')
    lm2.zero_grad()
    pnlty = 0.0
    lm2, err_train = Train_LinearModel_Mto1(lm2, M, pnlty, ftr_train_tnsr, trg_train_tnsr, lr, Max_steps=100000)

    print('valid')
    err_valid = Valid_LinearModel_Mto1(lm2, ftr_valid_tnsr, trg_valid_tnsr)

    y2tst_post = lm2.Predict(ftr_train_tnsr)
    ax_y2.plot(trg_train_tnsr.detach(), y2tst_post.detach(), 'bo', ms=7, mfc='none', label='post-training')
    ax_y2.legend()

def main_TorchnnLinearRegression_withGroundTruth():
    pass

    ### Ground Truth Data ###
    # fig_xx, ax_xx = plt.subplots()
    # ax_xx.set_title('XX')

    N = 40
    M = 5
    m = 3
    the_coefs=[0.0, 1.0, 0.0]
    the_bias = 0.0

    # ftrs, trgs = Polynomial_Data_Generator(N=N, M=M, m=m, coefs=the_coefs, cst=the_bias, x_low_bound=0.5, x_high_bound=8.5, noise=1.0)
    ftrs, trgs = Sine_Data_Generator(N=N, M=M, noise=0.0)

    X = torch.tensor(ftrs)
    Y = torch.tensor(trgs)
    print('X: ', X)
    print('Y: ', Y)
    # ax_xx.plot(X.detach(), Y.detach(), 'rx', ms=7)

    n_samples, in_dim = X.shape
    n_samples, out_dim = Y.shape
    print(f'n_samples = {n_samples}, n_features = {in_dim}')

    model = LinearRegressionFromNN(in_dim, out_dim)

    # sys.exit()
    ### Define loss and optimizer ###
    learning_rate = 1e-4

    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    ### Training loop ###
    max_step = 100000
    print_step = 10000
    print('Training Loop')
    for i in range(max_step):
        
        if i%print_step==0: print('forward')
        yprd = model(X)

        if i%print_step==0: print('loss')
        err = loss(Y, yprd)

        if i%print_step==0: print('derivative')
        err.backward()

        if i%print_step==0: print('grad_ws: ', model.lin.weight.grad )

        if i%print_step==0: print('update')
        optimizer.step()

        if i%print_step==0: print('zero out grad')
        optimizer.zero_grad()

        if i%print_step==0:
            w, b = model.parameters()
            print('i.{:d} / {:d}, loss = {:.3f}'.format(i, max_step, err.item()), 'ws: ', w.detach(), ' , b: ', b.detach(), ' grad_ws: ', model.lin.weight.grad )

    with torch.no_grad():
        Y_tst = model(X)
    fig, ax = plt.subplots()
    ax.set_title('test')
    ax.plot(Y.detach(), Y_tst.detach(), 'ro', ms=7, mfc='none', mew=1)
    ax.grid(True, which='both')
    
def main_TorchnnLinearRegression_withGroundTruth2(): # with training - testing separation
    pass

    N=200
    M=3
    ### polynomial data ###
    the_bias = 0.5
    the_coefs = [1,2,0]
    m=M
    ftrs, trgs = Polynomial_Data_Generator(N, M, m, the_coefs, the_bias)
    # ftrs, trgs = Sine_Data_Generator(N,M)
    num_trial, Data_Cluster = RandomSplit_CrossVal_Preparation(N, ftrs, trgs, set_num_trials=1.0, set_valid_ratio=0.1)

    ftr_train_tnsr = torch.tensor(Data_Cluster[0]['fit']['feature'])
    trg_train_tnsr = torch.tensor(Data_Cluster[0]['fit']['target'])
    ftr_valid_tnsr = torch.tensor(Data_Cluster[0]['valid']['feature'])
    trg_valid_tnsr = torch.tensor(Data_Cluster[0]['valid']['target'])

    # sys.exit()
    ### Training ###
    lr=1e-3
    lm = LinearRegressionFromNN(M, 1)
    # loss = nn.MSELoss()
    loss = lambda t,y: ((t-y)**2).mean()
    optimizer = torch.optim.SGD(lm.parameters(), lr=lr)

    with torch.no_grad():
        y_test = lm(ftr_train_tnsr)
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle('Prior Training, training data')
    ax1.plot(trg_train_tnsr.detach(), 'bo', ms=7, mfc='none', mew=1)
    ax1.plot(y_test.detach(), 'rx', ms=7, mfc='none', mew=1)
    ax2.set_xlabel('training target')
    ax2.set_ylabel('predict target')
    ax2.plot(trg_train_tnsr.detach(), y_test.detach(), 'ro', ms=7, mfc='none', mew=1)

    max_step = 10000
    print_step = 1000
    for i in range(max_step):
        
        yprd = lm(ftr_train_tnsr)

        err = loss(trg_train_tnsr, yprd)

        err.backward()

        optimizer.step()

        optimizer.zero_grad()

        if i%print_step==0:
            w, b = lm.parameters()
            print('i.{:d} / {:d}, loss = {:.3f}'.format(i, max_step, err.item()), 'ws: ', w.detach(), ' , b: ', b.detach(), ' grad_ws: ', lm.lin.weight.grad )

    with torch.no_grad():
        y_val = lm(ftr_valid_tnsr)

    figv, (axv1, axv2) = plt.subplots(1,2)
    figv.suptitle('Posterior Training, testing data')
    axv1.plot(trg_valid_tnsr.detach(), 'bo', ms=7, mfc='none', mew=1)
    axv1.plot(y_val.detach(), 'rx', ms=7, mfc='none', mew=1)
    axv2.set_xlabel('testing target')
    axv2.set_ylabel('predict target')
    axv2.plot(trg_valid_tnsr.detach(), y_val.detach(), 'ro', ms=7, mfc='none', mew=1)

    fig_ws, ax_ws = plt.subplots()
    fig_ws.suptitle('final weights')
    clrs = cm.rainbow(np.linspace(0,1,lm.lin.weight.shape[0]))
    for i in range(lm.lin.weight.shape[0]):
        ax_ws.plot(lm.lin.weight[i].detach(), 'ro', ms=7, mfc='none', mew=1.5, label='i.{:d}'.format(i))
    ax_ws.legend()

def main_TorchLinearRegression_Comparison_GroundTruth_CrossValid():
    pass 

    N=400
    M=5
    m=5
    the_coefs=[2.0, 1.5 ,0,0,0]
    the_bias = 1.0
    # features, tarts = Polynomial_Data_Generator(N, M, m, the_coefs, the_bias)
    features, tarts = Sine_Data_Generator(N,M,1)
    targets = np.array([item[0] for item in tarts]) 

    cross_validation_method = 'rand_cross'
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
    num_trial, Data_Cluster = glh_map.DataPreparation_for_CrossValidation(cross_validation_method, N, features, targets, parameter_crossvalidation)

    ### Training ###
    Penalties = np.logspace(-1,np.log10(3),11)

    ### Ridge MAP ### 
    print('Ridge MAP usual')
    R_Ridge = glh_map.Model_Training(num_trial, Data_Cluster, M, regularisation='ridge', Panelty_Values=Penalties)
    glh_map.Plot_One_Training_Result(R_Ridge)

    # sys.exit()
    ### Ridge MAP - PyTorch ###
    print('Ridge MAP PyTorch - Gradient descent')
    lr = 1e-6
    lm = LinaerModel_Mto1(M)

    Ave_Ws = []
    Std_Ws = []
    Ave_W0 = []
    Std_W0 = []
    Ave_Err = []
    Std_Err = []
    for i, pnlt in enumerate(Penalties):
        print('Pytorch pnlt idx.{:d} / {:d}'.format(i, len(Penalties)))
        ws_trials = []
        w0s_trials = []
        errs_trials = []
        for j in range(num_trial):
            print('Pytorch pnlt idx.{:d} / {:d}, trial{:d} / {:d}'.format(i, len(Penalties), j, num_trial))
            ftr_train_tnsr = torch.tensor(Data_Cluster[j]['fit']['feature'])
            trg_train_tnsr = torch.tensor(Data_Cluster[j]['fit']['target'])
            ftr_valid_tnsr = torch.tensor(Data_Cluster[j]['valid']['feature'])
            trg_valid_tnsr = torch.tensor(Data_Cluster[j]['valid']['target'])

            lm.zero_grad()
            lm, err_train = Train_LinearModel_Mto1(lm, M, pnlt, ftr_train_tnsr, trg_train_tnsr, lr, Max_steps=100000)
            err_valid = Valid_LinearModel_Mto1(lm, ftr_valid_tnsr, trg_valid_tnsr)

            ws_trials.append(np.copy(lm.Ws.detach().numpy()))
            w0s_trials.append(lm.W0.detach().item())
            errs_trials.append(err_valid.detach().item())
        
        ws_trials = np.array(ws_trials)
        w0s_trials = np.array(w0s_trials)
        errs_trials = np.array(errs_trials)

        Ave_Err.append(np.mean(errs_trials))
        Std_Err.append(np.std(errs_trials))
        Ave_W0.append(np.mean(w0s_trials))
        Std_W0.append(np.std(w0s_trials))
        Ave_Ws.append(np.mean(ws_trials, axis=0))
        Std_Ws.append(np.std(ws_trials, axis=0))

    Ave_Err = np.array(Ave_Err)
    Std_Err = np.array(Std_Err)

    opt_idx_torch = np.where(Ave_Err==np.min(Ave_Err))[0][0]

    ftrs_tnsr = torch.tensor(features)
    trgs_tnsr = torch.tensor(targets)

    lmf = LinaerModel_Mto1(M)
    for i in range(M):
        lmf.Ws.detach()[i] = Ave_Ws[opt_idx_torch][i]
    lmf.W0.detach()[0] = Ave_W0[opt_idx_torch]

    print(Ave_Ws[opt_idx_torch], ' : ', Ave_W0[opt_idx_torch])
    print(lmf.Ws, ' : ', lmf.W0)

    yprdf = lmf.Predict(ftrs_tnsr)
    fig_f, ax_f = plt.subplots()
    ax_f.set_title('final comparison, y vs targets')
    ax_f.plot(trgs_tnsr.detach(), yprdf.detach(), 'ro', ms=6, mfc='none', mew=1)
    ax_f.grid(True, which='both')

def main_standard_showcase():

    # Linear regression
    # f = w * x 
    # here : f = 2 * x

    # 0) Training samples, watch the shape!
    X = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float32)
    Y = torch.tensor([[2], [4], [6], [8], [10], [12], [14], [16]], dtype=torch.float32)

    n_samples, n_features = X.shape
    print(f'n_samples = {n_samples}, n_features = {n_features}')

    # 0) create a test sample
    X_test = torch.tensor([5], dtype=torch.float32)

    # 1) Design Model, the model has to implement the forward pass!

    # Here we could simply use a built-in model from PyTorch
    # model = nn.Linear(input_size, output_size)

    class LinearRegression(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(LinearRegression, self).__init__()
            # define different layers
            self.lin = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.lin(x)

    input_size, output_size = n_features, n_features

    model = LinearRegression(input_size, output_size)

    print(f'Prediction before training: f({X_test.item()}) = {model(X_test).item():.3f}')

    # 2) Define loss and optimizer
    learning_rate = 0.01
    n_epochs = 100

    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 3) Training loop
    for epoch in range(n_epochs):
        # predict = forward pass with our model
        y_predicted = model(X)

        # loss
        l = loss(Y, y_predicted)

        # calculate gradients = backward pass
        l.backward()

        # update weights
        optimizer.step()

        # zero the gradients after updating
        optimizer.zero_grad()

        if (epoch+1) % 10 == 0:
            w, b = model.parameters() # unpack parameters
            print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l.item())

    print(f'Prediction after training: f({X_test.item()}) = {model(X_test).item():.3f}')

if __name__ =='__main__':

    print('Pytorch_study2.py')
    # sys.exit()

    # sys.exit()
    # main_TorchnnLinearRegression_withGroundTruth()

    # sys.exit()
    # main_standard_showcase()

    sys.exit()
    ### Global Parameters ###
    data_std_cut = 1e-4 # for neglecting non-varying components in feature data - dr.load_data(..., std_cut=data_std_cut, ...) 

    ### Data Loading & pre-Processing ###
    home_path = '/Users/chenliu/'
    project_path = 'Research_Projects/SVM-SwapMC'
    training_data_path = 'DATA/Training_data'
    training_data_file = 'Cnf1.xy'
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
    num_trial, Data_Cluster = glh_map.DataPreparation_for_CrossValidation(cross_validation_method, N, features, targets, parameter_crossvalidation)

    ### MLP part ###
    fts_training = torch.tensor(Data_Cluster[0]['fit']['feature'])
    trg_training = torch.tensor(Data_Cluster[0]['fit']['target'])
    fts_valid = torch.tensor(Data_Cluster[0]['valid']['feature'])
    trg_valid = torch.tensor(Data_Cluster[0]['valid']['target'])

    Ws = torch.randn(M, requires_grad=True)
    W0 = torch.randn(1, requires_grad=True)

    # def forward(x):


    