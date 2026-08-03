# Main for Spherical Clustering

import math
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import davies_bouldin_score, silhouette_score, accuracy_score
from Spherical_Clustering_class import Spherical_Clustering


n_samples = [60,100,150]
n_features = [2,5,10]
n_centers = [3,5,7]

for m in n_samples:
    for n in n_features:
        for nc in n_centers:
            with open('clustering_experiments/dataset_'+str(m)+'_'+str(n)+'_'+str(nc)+'.txt', 'w') as f:
                f.write('Synthetic dataset for clustering with n_samples=' + str(m) + ', n_features=' + str(n) + ', n_centers=' + str(nc) + ' (make_blobs - cluster_std=1.0)\n\n')
                X, y = make_blobs(n_samples=m, centers=nc, n_features=n, cluster_std=1.0)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

                # Selection of values of hyperparameters by Grid Search
                l_par = [3,4,5,7]
                d_par = [0.2,0.3,0.5,0.7,1,1.5]
                eps_par = [0.2,0.3,0.5,0.7,1,1.5]
                selected_parameters = {'l':l_par,'d':d_par,'eps':eps_par}
                sc_grid = GridSearchCV(Spherical_Clustering(), selected_parameters, cv=5, verbose = 10, n_jobs = 10)
                sc_grid.fit(X_train)
                best_params = sc_grid.best_params_
                f.write('Best hyperparameters = ' + str(best_params) +'\n\n')

                # Spherical Clustering
                s_clust = Spherical_Clustering(l = best_params['l'], d = best_params['d'], eps = best_params['eps'])
                s_clust.fit(X)
                y_clust, r_stack_clust, c_stack_clust = s_clust.assign_labels(X)
                labels = (np.unique(y_clust)).tolist()

                if labels[0] != 0:
                    n_clust = len(labels)
                else:
                    n_clust = len(labels)-1

                X_no_out = []  # list of points that are not outliers
                y_no_out = []  # list of original labels of points that are not outliers
                X_outliers_idx = []  # list of indexes of points that are outliers
                y_clust_no_out = []  # list of labels of points that are not outliers
                for i in range(m):
                    if y_clust[i] != 0:
                        X_no_out.append(X[i])
                        y_no_out.append(y[i]+1)
                        y_clust_no_out.append(y_clust[i])
                    else:
                        X_outliers_idx.append(i)

                f.write('Results from clustering:\n')
                f.write('Centers stack = ' + str(c_stack_clust) + '\n')
                f.write('Radius stack = ' + str(r_stack_clust) + '\n')
                f.write('Number of clusters = ' + str(n_clust) + '\n')
                f.write('Number of outliers = ' + str(len(X_outliers_idx)) + '\n')

                # measures of clustering considering points without outliers
                DB_index = davies_bouldin_score(X_no_out, y_clust_no_out)
                SC_index = silhouette_score(X_no_out, y_clust_no_out)
                f.write('DB_index = ' + str(DB_index) + '\n')
                f.write('SC_index = ' + str(SC_index) + '\n')

                # accuracy score
                acc = accuracy_score(y_no_out, y_clust_no_out)
                f.write('Accuracy score = ' + str(acc) + '\n')

                f.write(str(m) + ' & ' + str(n) + ' & ' + str(n_clust) + ' & ' + str(round(DB_index, 3)) + ' & ' + str(round(SC_index, 3)) + ' & ' + str(round(acc, 3)) +'\\\\')


