import math
from curses.ascii import DC1

import matplotlib
from itertools import product
from tqdm import tqdm
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
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

with open('clustering_experiments/example_box_' + str(m) + '_' + str(n) + '_' + str(nc) + '.txt', 'w') as f:
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

    for l, d, eps in tqdm(parameter_grid,desc="Grid search",unit="configuration"):

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

            #Mapping the scores to [0,1] to compute their average
            SC_scaled = (SC_index+1)/2
            DB_scaled = 1/(1+DB_index)

            Avg_score =(SC_scaled + DB_scaled)/2

            if Avg_score > best_score:
                best_params = {'l':l, 'd':d, 'eps':eps}
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

figure, axes = plt.subplots()
colors = cm.rainbow(np.linspace(0,1,sph_clst.n_regions))
for reg, c in zip(sph_clst.regions,colors):
    axes.scatter(reg,np.zeros(len(reg)), facecolor='None', edgecolor=c)
axes.scatter(sph_clst.outliers, np.zeros(len(sph_clst.outliers)), facecolor='None', edgecolor='gray')
plt.savefig('clustering_experiments/sliding_box.pdf')

y_pca = np.zeros(m)
labels = list(range(sph_clst.n_regions+1))
for r_idx,l in zip(sph_clst.regions_idx,labels[1:]):
    for i in r_idx:
        y_pca[i] = l
l = 1
X_l = []
X_no_l = []
C_l = np.zeros(n)
for i in range(m):
    if y_pca[i] == l:
        X_l.append(X[i])
    else:
        X_no_l.append(X[i])
X_l = np.array(X_l)
X_no_l = np.array(X_no_l)
for i in range(X_l.shape[0]):
    for j in range(n):
        C_l[j] = np.mean(X_l[:,j])
distances_l = {}
for i in range(X_l.shape[0]):
    distances_l[i] = np.linalg.norm(C_l - X_l[i])
d_l_max = max(distances_l.values())
d_max = d_l_max + best_params['eps']
edge = 2*d_max
xmin = C_l[0]-d_max
ymin = C_l[1]-d_max
rectangle = patches.Rectangle((xmin,ymin),edge,edge,fill=False,edgecolor='black')

figure, axes = plt.subplots()
axes.scatter(X_no_l[:, 0], X_no_l[:, 1], facecolor='None', edgecolor='gray')
axes.scatter(X_l[:, 0], X_l[:, 1], facecolor='None', edgecolor=colors[0])
plt.gca().add_patch(rectangle)
plt.axis('equal')
plt.savefig('clustering_experiments/example_box.pdf')