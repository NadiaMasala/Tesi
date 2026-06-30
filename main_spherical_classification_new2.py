# Main 2 for Spherical Classifier with selection of class in the optimal sphere

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

# Splitting the dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Splitting training classes by their labels
A_train = []
B_train = []
for i in range(X_train.shape[0]):
    if y[i] == labels[0]:
        A_train.append(X_train[i])
    elif y[i] == labels[1]:
        B_train.append(X_train[i])
A_train = np.array(A_train)
B_train = np.array(B_train)

# Defining training classes centroids
cA_train = np.zeros(A_train.shape[1])
cB_train = np.zeros(B_train.shape[1])
for j in range(X_train.shape[1]):
    cA_train[j] = np.mean(A_train[:, j])
    cB_train[j] = np.mean(B_train[:, j])

dA_train = {}
for i in range(A_train.shape[0]):
    dA_train[i] = np.linalg.norm(cA_train - A_train[i])
dB_train = {}
for j in range(B_train.shape[0]):
    dB_train[j] = np.linalg.norm(cB_train - B_train[j])

dA_min = min(dA_train.values())
dA_max = max(dA_train.values())
dB_min = min(dB_train.values())
dB_max = max(dB_train.values())
d_min = max(dA_min, dB_min)
d_max = min(dA_max, dB_max)

epsilon_par = list(np.linspace(d_min,d_max,5))
minpts_par = [5, 10, 15]
# the following values of hyperparameters fit with the first criterion of selection of the class in the sphere
C1_par = list(np.linspace(1e-1, 1e+4, 4))  # then try 10
C2_par = list(np.linspace(1e-1, 1e+4, 4))
center_par = ['fixed','free']
selected_parameters = {'epsilon':epsilon_par, 'minpts':minpts_par, 'C1':C1_par, 'C2':C2_par, 'center':center_par}
sc_grid = GridSearchCV(New_Spherical_Classifier(), selected_parameters, cv=5, verbose = 10, n_jobs = 10 )
sc_grid.fit(X_train, y_train)
best_params = sc_grid.best_params_
print(best_params)

# Spherical Classification
sc = New_Spherical_Classifier(epsilon = best_params['epsilon'], minpts = best_params['minpts'], C1 = best_params['C1'], C2 = best_params['C2'], center = best_params['center'])
sc.fit(X_train, y_train)
print(sc.in_label_)
y_pred = sc.predict(X_train)
print(classification_report(y_train, y_pred))
y_pred = sc.predict(X_test)
print(classification_report(y_test, y_pred))
print(sc.c_)

# 2D Graphics
figure, axes = plt.subplots()
axes.scatter(A[:,0], A[:,1], facecolor = "none", edgecolor = "b", s = 50, label='A')
axes.scatter(B[:,0], B[:,1], facecolor = "none", edgecolor = "r", s = 50, label='B')
circle = plt.Circle((sc.c_[0], sc.c_[1]), sc.r_, color='black', fill=False)
axes.add_artist(circle)
axes.set_aspect(1)
axes.legend(loc='upper center')
all_x0 = np.concatenate((X[:,0],[sc.c_[0] - sc.r_, sc.c_[0] + sc.r_]))
all_x1 = np.concatenate((X[:,1],[sc.c_[1] - sc.r_, sc.c_[1] + sc.r_]))
axes.set_xlim(min(all_x0)-1,max(all_x0)+1)
axes.set_ylim(min(all_x1)-1,max(all_x1)+1)
plt.title("Spherical Classification")
plt.show()