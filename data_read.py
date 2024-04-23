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

from scipy import interpolate as interplt
from scipy.interpolate import interp1d
import sys
from datetime import datetime
import os.path
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import imp

### file reading function
def load_data(full_file_path, plot_mobility_map=bool, std_cut=1e-4):
    f_obj = open(full_file_path, 'r')
    all_lines = f_obj.readlines()
    f_obj.close()

    raw_data_by_lines = []
    num_elemnts_by_lines = []
    for item in all_lines:
        data_line = []
        for s in item.split():
            data_line.append(float(s))
        raw_data_by_lines.append(data_line)
        num_elemnts_by_lines.append(len(data_line))

    data_shape = []
    previous_length = 0
    previous_count = 0
    for i in range(len(num_elemnts_by_lines)):
        n_elmnts = num_elemnts_by_lines[i]
        if n_elmnts==previous_length :
            previous_count += 1
            if i == len(num_elemnts_by_lines)-1:
                data_shape.append([previous_count, previous_length])
        else: 
            if i>0: 
                data_shape.append([previous_count, previous_length])
            previous_length = n_elmnts
            previous_count = 1

    # print('data_read.load_data : data shape = ', data_shape)

    data_mtrx = np.array(raw_data_by_lines[1:])

    X_coor = data_mtrx[:,0]
    Y_coor = data_mtrx[:,1]
    Type = data_mtrx[:,2]
    Mblty = data_mtrx[:,3]
    Mblty_Norm = Mblty/np.max(Mblty)
    Features = data_mtrx[:,4:]
    Feature_Names = ['f{:d}'.format(i) for i in range(len(Features[0,:]))]

    Num_Particles = len(X_coor)
    Num_Features = len(Features[0,:])

    Normalized_Features = np.zeros(shape=shape(Features))
    Std_of_features = []
    Mean_of_features = []
    Zero_std_features=[]

    for i in range(len(Features[0,:])):
        std_data = np.std(Features[:,i])
        mean_data = np.mean(Features[:,i])
        Normalized_Features[:,i] = (Features[:,i] - mean_data)/((std_data!=0)*std_data + (std_data==0)*1)
        Std_of_features.append(std_data)
        Mean_of_features.append(mean_data)
        Zero_std_features.append((std_data==0)*1.0)
    
    Std_of_features = np.array(Std_of_features)
    Mean_of_features = np.array(Mean_of_features)

    Num_Features_Kept = Num_Features-int(np.sum(Zero_std_features))
    Clean_Feature_Names = []
    Norm_Clean_Features = [] 
    Clean_Features = []
    Clean_Std_Features = []
    Clean_Mean_Features = []
    for i in range(Num_Features):
        if Zero_std_features[i]==0 and Std_of_features[i]>std_cut:
            Clean_Features.append(Features[:,i])
            Norm_Clean_Features.append(Normalized_Features[:,i])
            Clean_Feature_Names.append(Feature_Names[i])
            Clean_Std_Features.append(Std_of_features[i])
            Clean_Mean_Features.append(Mean_of_features[i])
    Norm_Clean_Features = np.array(Norm_Clean_Features)
    Norm_Clean_Features = Norm_Clean_Features.T
    Clean_Features = np.array(Clean_Features)
    Clean_Features = Clean_Features.T
    Clean_Std_Features = np.array(Clean_Std_Features)
    Clean_Mean_Features = np.array(Clean_Mean_Features)

    if plot_mobility_map:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # x_norm = (X_coor - np.mean(X_coor))/np.std(X_coor)
        # y_norm = (Y_coor - np.mean(Y_coor))/np.std(Y_coor)
        # m_norm = (Mblty - np.mean(Mblty))/np.std(Mblty)
        ax.scatter( X_coor, Y_coor, Mblty, c=cm.coolwarm(Mblty_Norm) )

    return {'Num_Particles': Num_Particles, 'Num_Features': Num_Features, 'Num_Features_Clean': Num_Features_Kept, 'Types': Type,  'Coordinates': {'x': X_coor, 'y': Y_coor}, 'Mobility': Mblty, 'Raw_Features': Features, 'Raw_Feature_Names': Feature_Names, 'Norm_Clean_Features': Norm_Clean_Features, 'Clean_Feature_names': Clean_Feature_Names, 'Clean_Features': Clean_Features, 'Feature_Mean': Mean_of_features, 'Feanture_Std': Std_of_features, 'clean_std': Clean_Std_Features, 'clean_mean': Clean_Mean_Features}

def DataLoad_and_preProcessing(full_data_path, plot_mobility_map=0, std_cut=1e-4):
    
    all_data = load_data(full_data_path, plot_mobility_map=plot_mobility_map, std_cut=std_cut)

    X_coor = all_data['Coordinates']['x'] 
    Y_coor = all_data['Coordinates']['y'] 
    Mblty = all_data['Mobility']
    Types = all_data['Types']
    Clean_Features = all_data['Clean_Features']
    # Clean_Std_Features = all_data['clean_std']
    # Clean_Mean_Features = all_data['clean_mean']

    all_types = np.unique(Types)
    Mobility = {item: [] for item in all_types}
    Features = {item: [] for item in all_types}
    for i in range(len(Types)):
        p_type = Types[i]
        p_mobility = Mblty[i]
        p_features = Clean_Features[i]
        Mobility[p_type].append(p_mobility)
        Features[p_type].append(p_features)

    for item in Mobility.keys():
        Mobility[item] = np.array(Mobility[item])
        Features[item] = np.array(Features[item])

    select_type = all_types[0]
    targets = Mobility[select_type]
    features = Features[select_type]

    print('Number of total data points N : ', len(targets))
    print('Number of features M: ', len(features[0]))
    N = len(targets)
    M = len(features[0])

    std_features = np.array([np.std(features[:,i]) for i in range(M)])
    mean_features = np.array([np.mean(features[:,i]) for i in range(M)])

    return {'dimension of input feature vectors': M, 'total number of data points':N , 'input features': features, 'output targets': targets, 'empirical mean of features': mean_features, 'empirical standard deviation of features': std_features, 'feature names': all_data['Clean_Feature_names']}

def save_to_file_a_dictrionary(full_path, dict):
    asw = 'yes'
    if os.path.exists(full_path):
        asw = input('file path exists, overwrite? yes/no')
    if asw=='yes':
        np.save(full_path, dict)

def read_a_dictionary_file(full_path):
    return np.load(full_path+'.npy', allow_pickle=True).item()

def main():
    home_path = '/Users/chenliu/'
    project_path = 'Research_Projects/SVM-SwapMC'
    training_data_path = 'DATA/Training_data'
    training_data_file = 'Cnf1.xy'

    full_data_path = os.path.join(home_path, project_path, training_data_path, training_data_file )

    K = load_data(full_file_path=full_data_path, plot_mobility_map=True)
    print('main finished.')

if __name__ == "__main__":
    main()