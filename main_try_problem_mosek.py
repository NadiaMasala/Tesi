#Main for testing the implementation of the optimization problem in the mosek version

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs, make_circles, make_gaussian_quantiles
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from New_Spherical_Class_class import New_Spherical_Classifier
from New_Helper_SC import *


n_features = 2

# Creation of a casual dataset with 2 clusters
X, y = make_blobs(n_samples=100, centers=2, n_features=n_features, cluster_std=0.85)

m = X.shape[0]
n = X.shape[1]

labels = np.unique(y)

A = []
B = []
for i in range(m):
    if y[i] == labels[0]:
        A.append(X[i])
    elif y[i] == labels[1]:
        B.append(X[i])
A = np.array(A)
B = np.array(B)

figure, axes = plt.subplots()
axes.scatter(A[:,0], A[:,1], facecolor = "none", edgecolor = "b", s = 50, label='A')
axes.scatter(B[:,0], B[:,1], facecolor = "none", edgecolor = "r", s = 50, label='B')
axes.legend(loc='upper center')
plt.show()

barycenter = np.zeros(n)
for j in range(n):
    barycenter[j] = np.mean(X[:, j])
distances = {}
for i in range(m):
    distances[i] = np.linalg.norm(barycenter - X[i])
d_min = min(distances.values())
d_max = max(distances.values())

# Hyperparameters
epsilon = np.mean([d_min,d_max])
minpts = 3
C1 = 2e0
C2 = 2e0

# Optimization Problem
r_star, c_star, xi_in_star, xi_out_star, X_in, X_out, in_label, out_label = spherical_class_fit_semidef2_mosek(X, y, epsilon, minpts, C1, C2)
print('class in = '+str(in_label), 'class out = '+str(out_label), 'optimal center = '+str(c_star), 'optimal radius = '+str(r_star))