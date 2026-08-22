import math
from curses.ascii import DC1

import matplotlib
from itertools import product
from tqdm import tqdm
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

m = 50
nc = 3
n = 2

with open('clustering_experiments/clust_example_' + str(m) + '_' + str(n) + '_' + str(nc) + '.txt', 'w') as f:
    f.write('Synthetic dataset for clustering with n_samples=' + str(m) + ', n_features=' + str(n) + ', n_centers=' + str(nc) + ' (make_blobs - cluster_std=1.0)\n\n')

    X, y = make_blobs(n_samples=m, centers=nc, n_features=n, cluster_std=1.0)
    # Selection of values of hyperparameters by Grid Search
    l_par = [3, 4, 5, 7]
    d_par = [0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5]
    eps_par = [0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5]

    parameter_grid = list(product(l_par, d_par, eps_par))

    results = []

    best_score = 1e-2
    best_params = {'l': 0, 'd': 0, 'eps': 0}

    for l, d, eps in tqdm(parameter_grid, desc="Grid search", unit="configuration"):

        sph_clst = Spherical_Clustering(l=l, d=d, eps=eps)
        try:
            sph_clst.fit(X)
            y_clust, r_stack_clust, c_stack_clust = sph_clst.assign_labels(X)

            labels = (np.unique(y_clust)).tolist()

            if labels[0] != 0:
                n_clust = len(labels)
            else:
                n_clust = len(labels) - 1

            X_no_out = []  # list of points that are not outliers
            X_outliers_idx = []  # list of indexes of points that are outliers
            y_clust_no_out = []  # list of labels of points that are not outliers
            for i in range(m):
                if y_clust[i] != 0:
                    X_no_out.append(X[i])
                    y_clust_no_out.append(y_clust[i])
                else:
                    X_outliers_idx.append(i)

            DB_index = davies_bouldin_score(X_no_out, y_clust_no_out)
            SC_index = silhouette_score(X_no_out, y_clust_no_out)

            # Mapping the scores to [0,1] to compute their average
            SC_scaled = (SC_index + 1) / 2
            DB_scaled = 1 / (1 + DB_index)

            Avg_score = (SC_scaled + DB_scaled) / 2

            if Avg_score > best_score:
                best_params = {'l': l, 'd': d, 'eps': eps}
                best_score = Avg_score
        except:
            continue

    sph_clst = Spherical_Clustering(l=best_params['l'], d=best_params['d'], eps=best_params['eps'])
    sph_clst.fit(X)
    y_clust, r_stack_clust, c_stack_clust = sph_clst.assign_labels(X)

    f.write('Best hyperparameters = ' + str(best_params) + '\n\n')
    f.write('Results from clustering:\n')
    f.write('Centers stack = ' + str(c_stack_clust) + '\n')
    f.write('Radius stack = ' + str(r_stack_clust) + '\n')
    f.write('Number of clusters = ' + str(n_clust) + '\n')
    f.write('Number of outliers = ' + str(len(X_outliers_idx)) + '\n')
    f.write('DB_index = ' + str(DB_index) + '\n')
    f.write('SC_index = ' + str(SC_index) + '\n')

    f.write(str(m) + ' & ' + str(n) + ' & ' + str(n_clust) + ' & ' + str(round(DB_index, 3)) + ' & ' + str(round(SC_index, 3)) + ' & ' + str(round(best_score, 3)) + '\\\\')

if n == 2:
    figure, axes = plt.subplots()
    axes.scatter(X[:, 0], X[:, 1], c=y_clust)
    for l in sph_clst.labels:
        if l != 0:
            for j, k, c, r in zip(range(len(c_stack_clust)), range(len(r_stack_clust)), c_stack_clust, r_stack_clust):
                circle = plt.Circle((c[0], c[1]), r, color='black', fill=False)
                axes.add_artist(circle)
                axes.set_aspect(1)
    axes.set_xlim(-20, 20)
    axes.set_ylim(-20, 20)
    axes.set_box_aspect(1)
    #plt.title("Spherical Clustering - n_samples = " + str(m) + ", n_features = " + str(n) + ", n_clusters = " + str(n_clust))
    plt.axis('equal')
    plt.savefig('clustering_experiments/clust_example_' + str(m) + '_' + str(n) + '_' + str(nc) + 'fig.pdf')