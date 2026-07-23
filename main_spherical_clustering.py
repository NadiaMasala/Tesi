# Main for Spherical Clustering

import math
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from Spherical_Clustering_class import Spherical_Clustering

m = 20
n = 2
X, y = make_blobs(n_samples=m, centers=3, n_features=n, cluster_std=0.8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Selection of values of hyperparameters by Grid Search
l_par = [3,4,5]
d_par = [0.3,0.5,1]
C1_par = list(np.linspace(1e-1, 1e+4, 4))
C2_par = list(np.linspace(1e-1, 1e+4, 4))
center_par = ['fixed','free']
eps_par = [1,3,5]
selected_parameters = {'l':l_par,'d':d_par,'C1':C1_par, 'C2':C2_par, 'center':center_par, 'eps':eps_par}
sc_grid = GridSearchCV(Spherical_Clustering(), selected_parameters, cv=5, verbose = 10, n_jobs = 10)
sc_grid.fit(X_train, y_train)
best_params = sc_grid.best_params_
print('Best hyperparameters = '+ str(best_params) + '\n')

# Spherical Clustering
sc = Spherical_Clustering(l = best_params['l'], d = best_params['d'],C1 = best_params['C1'], C2 = best_params['C2'], center = best_params['center'], eps = best_params['eps'])
sc.fit(X)



# Graphics
if n == 2:
    figure, axes = plt.subplots()
    axes.scatter(A[:, 0], A[:, 1], facecolor="none", edgecolor="b", s=50, label='A')
    axes.scatter(B[:, 0], B[:, 1], facecolor="none", edgecolor="r", s=50, label='B')
    circle = plt.Circle((sc.c_[0], sc.c_[1]), sc.r_, color='black', fill=False)
    axes.add_artist(circle)
    axes.set_aspect(1)
    all_x0 = np.concatenate((X[:, 0], [sc.c_[0] - sc.r_, sc.c_[0] + sc.r_]))
    all_x1 = np.concatenate((X[:, 1], [sc.c_[1] - sc.r_, sc.c_[1] + sc.r_]))
    axes.set_xlim(min(all_x0) - 1, max(all_x0) + 1)
    axes.set_ylim(min(all_x1) - 1, max(all_x1) + 1)
    plt.title("Spherical Classification - n_samples = "+str(ns)+", n_features = "+str(nf))
    plt.savefig('experiments/fig_mb_'+str(ns)+'_'+str(nf)+'.pdf')
    #plt.savefig('experiments/fig_mc_'+str(ns)+'_'+str(nf)+'.pdf')

