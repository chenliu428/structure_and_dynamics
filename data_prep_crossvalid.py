import math as ma
from numpy import *
from pylab import *
from scipy import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys
import os.path
import imp


def Creat_Nest(features, targets, num_cell, print_info:bool=False):
    num_data_points_per_cell = int(len(targets)/num_cell)

    Nest = []
    for i in range(num_cell):
        validation_features = features[i*num_data_points_per_cell:(i+1)*num_data_points_per_cell]
        validation_targets = targets[i*num_data_points_per_cell:(i+1)*num_data_points_per_cell]
        fitting_features = np.append(features[:i*num_data_points_per_cell], features[(i+1)*num_data_points_per_cell:], axis=0)
        fitting_targets = np.append(targets[:i*num_data_points_per_cell], targets[(i+1)*num_data_points_per_cell:])
        Nest.append({'fit':{'feature':fitting_features, 'target':fitting_targets}, 'valid':{'feature':validation_features, 'target':validation_targets}})

    if print_info:
        print('Creat_Nest: ')
        for i in range(len(Nest)):
            item = Nest[i]
            print('{:d}-th trial: fit - '.format(i), shape(item['fit']['feature']), shape(item['fit']['target']), ' valid - ', shape(item['valid']['feature']), shape(item['valid']['target']))
    
    return Nest

def Data_Segmentation(N:int, val_portion, num_seg:int, features, targets, rand_seed=323242323, rand_seeding:bool=False, print_info:bool=False):
    Nf = len(features[:,0])
    Nt = len(targets)
    M = len(features[0,:])
    if N!=Nt or N!=Nf:
        print('wrong size - - !')
        sys.exit()
    
    val_data_num = int(N*val_portion)
    train_data_num = int(N - val_data_num)
    num_points_per_seg = int(train_data_num/num_seg)
    num_remainders = train_data_num%num_seg

    indexs = [i for i in range(N)]

    if rand_seeding: np.random.seed(rand_seed)
    
    val_data = {'feature': [], 'target': []}
    for i in range(val_data_num):
        idx = int(np.random.uniform(0,len(indexs)))
        data_idx = indexs[idx]
        val_data['feature'].append(features[data_idx,:])
        val_data['target'].append(targets[data_idx])
        indexs.pop(idx)
    val_data['feature']=np.array(val_data['feature'])
    val_data['target']=np.array(val_data['target'])

    Segments=[]
    for i in range(num_seg):
        seg = []
        num_this_seg = num_points_per_seg + 1*(i<num_remainders)
        while len(seg)<num_this_seg:
            idx = int(np.random.uniform(0,len(indexs)))
            seg.append(indexs[idx])
            indexs.pop(idx)
        Segments.append(seg)

    Data_Sgmts = []
    for i in range(len(Segments)):
        data={'feature': [], 'target': []}
        for idx in Segments[i]:
            data['target'].append(targets[idx])
            data['feature'].append(features[idx,:])
        data['feature'] = np.array(data['feature'])
        data['target'] = np.array(data['target'])
        Data_Sgmts.append({'fit':data, 'valid':val_data})

    if print_info:
        print('Data_Segmentation: ')
        for i in range(len(Data_Sgmts)):
            print('{:d}-th segment: fit -- '.format(i), shape(Data_Sgmts[i]['fit']['feature']), shape(Data_Sgmts[i]['fit']['target']), ' valid -- '.format(i), shape(Data_Sgmts[i]['valid']['feature']), shape(Data_Sgmts[i]['valid']['target']))
    
    return Data_Sgmts


def Random_Splits(N, val_data_portion, Num_RandCross, features, targets, rand_seed=467237746, rand_seeding:bool=True, print_info:bool=False):
    Data_RandCross = []
    if rand_seeding: np.random.seed(rand_seed)
    for i in range(Num_RandCross):
        data_chuck = Data_Segmentation(N, val_data_portion, num_seg=1, features=features, targets=targets, rand_seeding=False, print_info=False)
        if len(data_chuck)!=1:
            print('sth wrong')
            sys.exit()
        Data_RandCross.append(data_chuck[0])

    if print_info:
        print('Random_Splits: ')
        for i in range(len(Data_RandCross)):
            print('{:d}-th segment: fit -- '.format(i), shape(Data_RandCross[i]['fit']['feature']), shape(Data_RandCross[i]['fit']['target']), ' valid -- '.format(i), shape(Data_RandCross[i]['valid']['feature']), shape(Data_RandCross[i]['valid']['target']))
    return Data_RandCross

def DataPreparation_for_CrossValidation(cross_validation_method, N, features, targets, parameters:dict):
    rand_seed = parameters['rand_seed']
    seeding = parameters['seeding_or_not']
    print_info = parameters['print_info_or_not']
    if cross_validation_method == 'nest':
        num_cell = parameters['num_cell'] #int(5) # number of cells
        Nest = Creat_Nest(features, targets, num_cell, print_info=print_info)
        num_trial = num_cell
        Data_Cluster = Nest
    elif cross_validation_method == 'segment':
        val_data_portion = parameters['validation_ratio']
        num_seg = parameters['num_seg'] # int(3)
        # rand_seed = 473846332
        # np.random.seed(rand_seed)
        Data_Sgmts = Data_Segmentation(N, val_data_portion, num_seg=num_seg, features=features, targets=targets, rand_seed=rand_seed, rand_seeding=seeding, print_info=print_info)
        num_trial = num_seg 
        Data_Cluster = Data_Sgmts
    elif cross_validation_method == 'rand_cross':
        val_data_portion = parameters['validation_ratio']
        Num_RandCross = int(parameters['num_split_ratio']/val_data_portion)
        # rand_seed = 473846332
        # np.random.seed(rand_seed)
        Data_RandCross = Random_Splits(N, val_data_portion, Num_RandCross, features, targets, rand_seed=rand_seed, rand_seeding=seeding, print_info=print_info)
        num_trial = Num_RandCross
        Data_Cluster = Data_RandCross
    else:
        print('non existing cross_validation_method : ', cross_validation_method)
        sys.exit()
    return num_trial, Data_Cluster