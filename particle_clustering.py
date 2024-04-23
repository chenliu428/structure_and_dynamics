import math as ma
from numpy import *
from pylab import *
from scipy import *
import os.path
from matplotlib import rc, rcParams
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
from matplotlib.colors import Normalize

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

import sklearn.cluster as skcltr

import data_read as dr
imp.reload(dr)

def main_propensity_map(Cnf_name:str='Cnf2.xy', draw_cluster:bool=False):

    home_path = '/Users/chenliu/'
    project_path = 'Research_Projects/SVM-SwapMC'
    training_data_path = 'DATA/Training_data'
    training_data_file = Cnf_name

    full_data_path = os.path.join(home_path, project_path, training_data_path, training_data_file)

    all_data = dr.load_data(full_data_path, plot_mobility_map=0)

    X_coor = all_data['Coordinates']['x'] 
    Y_coor = all_data['Coordinates']['y'] 
    Mblty = all_data['Mobility']
    Mblty_Norm = Mblty/np.max(Mblty)

    x_norm = (X_coor - np.mean(X_coor))/np.std(X_coor)
    y_norm = (Y_coor - np.mean(Y_coor))/np.std(Y_coor)
    m_norm = (Mblty - np.mean(Mblty))/np.std(Mblty)

    x_new =[]
    y_new =[]
    m_new = []
    for i in range(len(x_norm)):
        if x_norm[i]>-100.25 and y_norm[i]<100.5:
            x_new.append(x_norm[i])
            y_new.append(y_norm[i])
            m_new.append(m_norm[i])
    x_new = np.array(x_new)
    y_new = np.array(y_new)
    m_new = np.array(m_new)

    # print(len(x_new))

    AggCluster = skcltr.AgglomerativeClustering(n_clusters=2, metric='euclidean', connectivity=None, compute_full_tree=True, linkage='complete')

    X_input = np.array([x_new, y_new, m_new]).T

    AggCluster.fit(X_input)
    clusters = copy(AggCluster.labels_)
    labels = np.unique(clusters)
    most_populated_label = labels[0]
    most_population = 0
    for item in labels:
        population_item = np.sum(1.0*(clusters==item))
        most_populated_label  = item if (population_item>=most_population) else most_populated_label
        most_population = population_item if (population_item>=most_population) else most_population

    clrmap = cm.coolwarm

    fig_m, ax_m = plt.subplots(figsize=(7,6))
    ax_m.set_xlabel('x')
    ax_m.set_ylabel('y')
    fig_m.suptitle('Propensity map')
    for i in range(len(x_new)):
        ax_m.plot([x_new[i]],[y_new[i]], 'o', mec='none', mfc=clrmap(Mblty_Norm[i]), ms=8)

    if draw_cluster:
        for i in range(len(x_new)):
            digit = clusters[i]
            # if digit in [2,5,7,8,10,11]: 
            if digit != most_populated_label:
                pass
                # plt.plot([x_new[i]],[y_new[i]], marker=f"${digit}$", mec='none', mfc='k', ms=5)
                plt.plot([x_new[i]],[y_new[i]], 'o', mec='k', mew=0.5, mfc='none', ms=8)

    avMblty_of_Labels = []
    for i in range(len(labels)):
        avm = np.sum((clusters==labels[i])*m_new)/np.sum(clusters==labels[i])
        avMblty_of_Labels.append(avm)

    fig_m.colorbar(plt.cm.ScalarMappable(norm=Normalize(np.min(Mblty), np.max(Mblty)), cmap=clrmap), ax=ax_m, label="Propensity")

    mb_sort = copy(avMblty_of_Labels)
    mb_sort.sort()
    lb_sort = []
    for item in mb_sort:
        idx=avMblty_of_Labels.index(item)
        lb_sort.append(labels[idx])
    lb_sort=np.array(lb_sort)

    # plt.figure()
    # plt.plot(mb_sort, 'bX')

if __name__ == '__main__': 
    main_propensity_map('Cnf2.xy')