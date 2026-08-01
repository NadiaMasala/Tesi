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
from sklearn.metrics import davies_bouldin_score, silhouette_score
from Spherical_Clustering_class import Spherical_Clustering


m = 40
n = 3

with open('clustering/dataset_mb_new22_'+str(m)+'_'+str(n)+'.txt', 'w') as f:
#with open('clustering/dataset_mc_new22_' + str(m) + '_' + str(n) + '.txt', 'w') as f:
    f.write('Synthetic dataset for clustering with n_samples=' + str(m) + ' and n_features=' + str(n) + '\n(make_blobs - cluster_std=1.4)\n')
    X, y = make_blobs(n_samples=m, centers=3, n_features=n, cluster_std=1.4)  # cluster_std=0.8 for perfectly separable clusters

    #f.write('Synthetic dataset for clustering with n_samples=' + str(m) + ' and n_features=' + str(n) + '\n(make_classification - class_sep=1.3)\n')
    #X, y = make_classification(m, n, n_classes=3, n_clusters_per_class=1, class_sep=1.3, n_informative=n, n_redundant=0, n_repeated=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Selection of values of hyperparameters by Grid Search
    l_par = [3,4,5]
    d_par = [0.2,0.3,0.5]
    #eps_par = [0.2,0.3,0.5]
    eps_par = [0.5, 1, 2]
    selected_parameters = {'l':l_par,'d':d_par,'eps':eps_par}
    sc_grid = GridSearchCV(Spherical_Clustering(), selected_parameters, cv=5, verbose = 10, n_jobs = 10)
    sc_grid.fit(X_train)
    best_params = sc_grid.best_params_
    f.write('Best hyperparameters = ' + str(best_params) + '(eps_par = [0.5, 1, 2])' +'\n')

    # Spherical Clustering
    s_clust = Spherical_Clustering(l = best_params['l'], d = best_params['d'], eps = best_params['eps'])
    s_clust.fit(X)
    c_stack = s_clust.c_stack
    r_stack = s_clust.r_stack
    y_clust = s_clust.assign_labels(X)
    labels = (np.unique(y_clust)).tolist()

    if labels[0] != 0:
        n_clust = len(labels)
    else:
        n_clust = len(labels)-1

    n_regions = s_clust.n_regions
    regions = s_clust.regions
    outliers = s_clust.outliers

    X_in_clusters = []  # list of points in the clusters
    X_outliers = []  # list of points that are outliers
    for i in range(m):
        if y_clust[i] != 0:
            X_in_clusters.append(X[i])
        else:
            X_outliers.append(X[i])
    y_clust_no_out = []  # list of labels of points that are not outliers
    for i in range(m):
        if y_clust[i] != 0:
            y_clust_no_out.append(y_clust[i])

    f.write('Centers stack = ' + str(c_stack) + '\n')
    f.write('Radius stack = ' + str(r_stack) + '\n')
    f.write('Number of clusters = ' + str(n_clust) + '\n')

    # measures of clustering considering all points
    f.write('Measures of clustering considering all points: \n')
    DB_index = davies_bouldin_score(X,y_clust)
    SC_index = silhouette_score(X,y_clust)
    f.write('DB_index = ' + str(DB_index) + '\n')
    f.write('SC_index = ' + str(SC_index) + '\n')
    # measures of clustering considering points without outliers
    f.write('Measures of clustering considering points without outliers: \n')
    DB_index1 = davies_bouldin_score(X_in_clusters, y_clust_no_out)
    SC_index1 = silhouette_score(X_in_clusters, y_clust_no_out)
    f.write('DB_index = ' + str(DB_index1) + '\n')
    f.write('SC_index = ' + str(SC_index1) + '\n')

    f.write(str(m) + ' & ' + str(n) + ' & ' + str(n_clust) + ' & ' + str(round(DB_index, 3)) + ' & ' + str(round(SC_index, 3)) + '\\\\ \n')
    f.write(str(m) + ' & ' + str(n) + ' & ' + str(n_clust) + ' & ' + str(round(DB_index1, 3)) + ' & ' + str(round(SC_index1, 3)) + '\\\\')

# Sliding Window graphic
figure, axes = plt.subplots()
colors = cm.rainbow(np.linspace(0,1,n_regions+1))
for reg, c in zip(regions,colors[1:]):
    axes.scatter(reg,np.zeros(len(reg)), facecolor='none', edgecolor=c)
if len(outliers) > 0:
    axes.scatter(outliers, np.zeros(len(outliers)), facecolor='none', edgecolor=colors[0])
plt.savefig('clustering/sw_mb_new22_' + str(m) + '_' + str(n) + '_' + str(s_clust.n_regions) + '.pdf')
#plt.savefig('clustering/sw_mc_new22_' + str(m) + '_' + str(n) + '_' + str(s_clust.n_regions) + '.pdf')
plt.show()

# Graphics
if n == 2:
    figure, axes = plt.subplots()
    axes.scatter(X[:, 0], X[:, 1], c=y_clust)
    for c, r in zip(c_stack, r_stack):
        circle = plt.Circle((c[0], c[1]), r, color='black', fill=False)
        axes.add_artist(circle)
        axes.set_aspect(1)
    axes.set_xlim(-20, 20)
    axes.set_ylim(-20, 20)
    axes.set_box_aspect(1)
    plt.title("Spherical Clustering - n_samples = "+str(m)+", n_features = "+str(n)+", n_clusters = "+str(n_clust))
    #plt.savefig('clustering/fig_clust2D_mc_new22_'+str(m)+'_'+str(n)+'_'+str(n_clust)+ '_' +'.pdf')
    plt.savefig('clustering/fig_clust2D_mb_new22_'+str(m)+'_'+str(n)+'_'+str(n_clust)+ '_' +'.pdf')
    plt.show()

    figure, axes = plt.subplots()
    axes.scatter(X[:, 0], X[:, 1], c=y_clust)
    for l in labels:
        if l != 0:
            for j, k, c, r in zip(range(len(c_stack)), range(len(r_stack)),c_stack, r_stack):
                if j == l-1 and k == l-1:
                    circle = plt.Circle((c[0], c[1]), r, color='black', fill=False)
                    axes.add_artist(circle)
                    axes.set_aspect(1)
    axes.set_xlim(-20, 20)
    axes.set_ylim(-20, 20)
    axes.set_box_aspect(1)
    plt.title(
        "Spherical Clustering - n_samples = " + str(m) + ", n_features = " + str(n) + ", n_clusters = " + str(n_clust))
    #plt.savefig('clustering/fig2_clust2D_mc_new22_'+str(m)+'_'+str(n)+'_'+str(n_clust)+ '_' +'.pdf')
    plt.savefig('clustering/fig2_clust2D_mb_new22_' + str(m) + '_' + str(n) + '_' + str(n_clust) + '_' + '.pdf')
    plt.show()

elif n == 3:
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    axes.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=y_clust)
    # Parametrization of the spheres
    theta = np.linspace(0, 2 * np.pi, 20)
    phi = np.linspace(0, np.pi, 20)
    for c, r in zip(c_stack, r_stack):
        x = c[0] + r * np.outer(np.sin(phi), np.cos(theta))
        y = c[1] + r * np.outer(np.sin(phi), np.sin(theta))
        z = c[2] + r * np.outer(np.cos(phi), np.ones_like(theta))
        # 3D graphic
        axes.plot_wireframe(x, y, z, color='k', linewidth=0.5)
        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_zlabel('z')
    axes.set_xlim(-20, 20)
    axes.set_ylim(-20, 20)
    axes.set_zlim(-20, 20)
    axes.set_box_aspect([1, 1, 1])
    plt.title(
        "Spherical Clustering - n_samples = " + str(m) + ", n_features = " + str(n) + ", n_clusters = " + str(n_clust))
    #plt.savefig('clustering/fig_clust3D_mc_new22_' + str(m) + '_' + str(n) + '_' + str(n_clust) + '.pdf')
    plt.savefig('clustering/fig_clust3D_mb_new22_' + str(m) + '_' + str(n) + '_' + str(n_clust) + '.pdf')
    plt.show()

    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    axes.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=y_clust)
    # Parametrization of the spheres
    theta = np.linspace(0, 2 * np.pi, 20)
    phi = np.linspace(0, np.pi, 20)
    for l in labels:
        if l != 0:
            for j, k, c, r in zip(range(len(c_stack)), range(len(r_stack)), c_stack, r_stack):
                if j == l - 1 and k == l - 1:
                    x = c[0] + r * np.outer(np.sin(phi), np.cos(theta))
                    y = c[1] + r * np.outer(np.sin(phi), np.sin(theta))
                    z = c[2] + r * np.outer(np.cos(phi), np.ones_like(theta))
                    # 3D graphic
                    axes.plot_wireframe(x, y, z, color='k', linewidth=0.5)
                    axes.set_xlabel('x')
                    axes.set_ylabel('y')
                    axes.set_zlabel('z')
    axes.set_xlim(-20, 20)
    axes.set_ylim(-20, 20)
    axes.set_zlim(-20, 20)
    axes.set_box_aspect([1, 1, 1])
    plt.title("Spherical Clustering - n_samples = " + str(m) + ", n_features = " + str(n) + ", n_clusters = " + str(n_clust))
    #plt.savefig('clustering/fig2_clust3D_mc_new22_' + str(m) + '_' + str(n) + '_' + str(n_clust) + '.pdf')
    plt.savefig('clustering/fig2_clust3D_mb_new22_' + str(m) + '_' + str(n) + '_' + str(n_clust) + '.pdf')
    plt.show()

    # i plot hanno tutte le sfere della classificazione binaria 1vsall,
    # fare (un altro) plot con le sole sfere relative alle etichette finali effettive ottenute con il clustering?



