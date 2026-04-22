#Main for testing the implementation of the optimization problem in the pyomo version

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
X, y = make_blobs(n_samples=100, centers=2, n_features=n_features, cluster_std=0.6)

m = X.shape[0]
n = X.shape[1]

barycenter = np.zeros(n)
for j in range(n):
    barycenter[j] = np.mean(X[:, j])

distances = {}
for i in range(m):
    distances[i] = np.linalg.norm(barycenter - X[i])

d_min = min(distances.values())
d_max = max(distances.values())

epsilon = np.mean([d_min,d_max])
C1 = 1e-4
C2 = 1e-4

r, xi_in, xi_out, X_in, X_out, in_label, out_label = new_spherical_class_fit_semidef_pyomo(X, y, epsilon, C1, C2)
print('class in = '+str(in_label), 'class out = '+str(out_label), 'optimal radius = '+str(r), 'optimal xi_in = '+str(xi_in), 'optimal xi_out = '+str(xi_out))