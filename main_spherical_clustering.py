# Main for Spherical Clustering

import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import davies_bouldin_score, silhouette_score
from Spherical_Clustering_class import Spherical_Clustering

m = 100
n = 2
X, y = make_blobs(n_samples=m, centers=3, n_features=n, cluster_std=0.8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Selection of values of hyperparameters by Grid Search
l_par = [3,4,5]
d_par = [0.2,0.3,0.5]
eps_par = [0.3,0.5,1]
selected_parameters = {'l':l_par,'d':d_par,'eps':eps_par}
sc_grid = GridSearchCV(Spherical_Clustering(), selected_parameters, cv=5, verbose = 10, n_jobs = 10)
sc_grid.fit(X_train)
best_params = sc_grid.best_params_
print('Best hyperparameters = '+ str(best_params) + '\n')

# Spherical Clustering
s_clust = Spherical_Clustering(l = best_params['l'], d = best_params['d'], eps = best_params['eps'])
s_clust.fit(X)
c_stack = s_clust.c_stack
print(c_stack)
r_stack = s_clust.r_stack
print(r_stack)
y_clust = s_clust.assign_labels(X)
print(y_clust)
labels = np.unique(y_clust)
print(labels)
n_clust = len(labels)-1

figure, axes = plt.subplots()
plt.scatter(X[:,0],X[:,1],c=y, s=50)
for c,r in zip(c_stack,r_stack):
    circle = plt.Circle((c[0], c[1]), r, color='black', fill=False)
    axes.add_artist(circle)
    axes.set_aspect(1)
plt.show()

'''DB_index = davies_bouldin_score(X,y)
print(DB_index)
SC_index = silhouette_score(X,y)
print(SC_index)'''

X_labeled = [[] for _ in range(len(labels))]
for l,lab in zip(range(len(labels)),labels):
    for i in range(m):
        if y[i] == lab:
            X_labeled[l].append(X[i])

#Sliding Window graphic
figure, axes = plt.subplots()
colors = cm.rainbow(np.linspace(0, 1, s_clust.n_regions))
for reg, c in zip(s_clust.regions, colors):
    axes.scatter(reg, np.zeros(len(reg)), facecolor='none', edgecolor=c, s=50)
axes.scatter(s_clust.outliers, np.zeros(len(s_clust.outliers)), facecolor='none', edgecolor='gray',s=50)
plt.show()
quit()
plt.savefig('clustering/sw_' + str(m) + '_' + str(n) + '_' + str(s_clust.n_regions) + '.pdf')
# Graphics
if n == 2:
    figure, axes = plt.subplots()
    colors = cm.rainbow(np.linspace(0, 1, len(labels)-1))
    for l, col in zip(labels[1:], colors):
        axes.scatter(X_labeled[l][:,0], X_labeled[l][:,1], facecolor="none", edgecolor=col, s=50)
    axes.scatter(X_labeled[0][:,0], X_labeled[0][:,1], facecolor="none", edgecolor="gray", s=50)
    for c,r in zip(c_stack,r_stack):
        circle = plt.Circle((c[0], c[1]), r, color='black', fill=False)
        axes.add_artist(circle)
        axes.set_aspect(1)
    plt.title("Spherical Clustering - n_samples = "+str(m)+", n_features = "+str(n)+", n_clusters = "+str(n_clust))
    plt.savefig('clustering/fig_clust2D_'+str(m)+'_'+str(n)+'_'+str(n_clust)+'.pdf')
elif n == 3:
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    colors = cm.rainbow(np.linspace(0, 1, len(labels)-1))
    for l,col in zip(labels[1:],colors):
        axes.scatter(X_labeled[l][:,0], X_labeled[l][:,1], X_labeled[l][:,2], facecolor="none", edgecolor=col, s=50)
    axes.scatter(X_labeled[0][:,0], X_labeled[0][:,1], X_labeled[0][:,2], facecolor="none", edgecolor="gray", s=50)
    # Parametrization of the spheres
    theta = np.linspace(0, 2 * np.pi, 20)
    phi = np.linspace(0, np.pi, 20)
    for c,r in zip(c_stack,r_stack):
        x = c[0] + r * np.outer(np.sin(phi), np.cos(theta))
        y = c[1] + r * np.outer(np.sin(phi), np.sin(theta))
        z = c[2] + r * np.outer(np.cos(phi), np.ones_like(theta))
        # 3D graphic
        axes.plot_surface(x,y,z, color='white', edgecolor='lightblue', alpha=0.3)
        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_zlabel('z')
    plt.title("Spherical Clustering - n_samples = "+str(m)+", n_features = "+str(n)+", n_clusters = "+str(n_clust))
    plt.savefig('clustering/fig_clust3D_'+str(m)+'_'+str(n)+'_'+str(n_clust)+'.pdf')


