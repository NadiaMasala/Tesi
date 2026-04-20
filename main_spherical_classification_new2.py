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
#y = label_binarize(y, classes=[0,1], neg_label=2, pos_label=7)

m = X.shape[0]
n = X.shape[1]

labels = np.unique(y)

# Splitting the points by their labels
A = []
B = []
for i in range(m):
    if y[i] == labels[0]:
        A.append(X[i])
    elif y[i] == labels[1]:
        B.append(X[i])
A = np.array(A)
B = np.array(B)

# Defining classes centroids
C_a = np.zeros(n)
C_b = np.zeros(n)
for j in range(n):
    C_a[j] = np.mean(A[:,j])
    C_b[j] = np.mean(B[:,j])

# Defining misclassified points to add noise in the dataset
D_a = np.zeros(A.shape[0])
D_b = np.zeros(B.shape[0])
for i in range(A.shape[0]):
    D_a[i] = np.linalg.norm(C_a - A[i])
R_a = np.max(D_a)
for j in range(B.shape[0]):
    D_b[j] = np.linalg.norm(C_b - B[j])
R_b = np.max(D_b)
A_extra = np.zeros((math.floor(A.shape[0]*(15/100)), n))
B_extra = np.zeros((math.floor(B.shape[0]*(15/100)), n))
for h in range(A_extra.shape[0]):
    for k in range(A_extra.shape[1]):
        A_extra[h,k] = np.random.uniform(C_a[k]-R_a, C_a[k]+R_a)
for h in range(B_extra.shape[0]):
    for k in range(B_extra.shape[1]):
        B_extra[h,k] = np.random.uniform(C_b[k]-R_b, C_b[k]+R_b)
y_A_extra = []
for h in range(A_extra.shape[0]):
    y_A_extra.append(labels[1])
y_A_extra = np.array(y_A_extra)
y_B_extra = []
for h in range(B_extra.shape[0]):
    y_B_extra.append(labels[0])
y_B_extra = np.array(y_B_extra)

# Creating new dataset with noise
X = np.concatenate((X, A_extra, B_extra))
y = np.concatenate((y.ravel(), y_A_extra, y_B_extra))
A = np.concatenate((A, B_extra))
B = np.concatenate((B, A_extra))

# Original plot 2D
#figure, axes = plt.subplots()
#axes.scatter(A[:,0], A[:,1], facecolor = "None", edgecolor = "b", s = 50, label='A')
#axes.scatter(B[:,0], B[:,1], facecolor = "None", edgecolor = "r", s = 50, label='B')
#plt.show()

# New dataset dimensions
m = X.shape[0]
n = X.shape[1]

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
minpts_par = [3, 5, 10, 15, 20]
#C1_par = [2, 5, 10, 15, 20]  # come scelgo i valori che possono assumere gli iperparametri?
# or try inside the interval [1e-4, 1e+4]
C1_par = list(np.linspace(1e-4, 1e+4, 5))
#C2_par = [2, 5, 10, 15, 20]  # come scelgo i valori che possono assumere gli iperparametri?
# or try inside the interval [1e-4, 1e+4]
C2_par = list(np.linspace(1e-4, 1e+4, 5))
center_par = ['fixed','free']
selected_parameters = {'epsilon':epsilon_par, 'minpts':minpts_par, 'C1':C1_par, 'C2':C2_par, 'center':center_par}
sc_grid = GridSearchCV(New_Spherical_Classifier(), selected_parameters, cv=10, verbose = 10, n_jobs = 10 )
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

# Graphics
if n == 2:
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
elif n == 3:
    matplotlib.use('TkAgg')
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    axes.scatter(A[:, 0], A[:, 1], A[:, 2], facecolor="none", edgecolor="b", s=50, label='A')
    axes.scatter(B[:, 0], B[:, 1], B[:, 2], facecolor="none", edgecolor="r", s=50, label='B')
    theta = np.linspace(0, 2 * np.pi, 20)
    phi = np.linspace(0, np.pi, 20)
    x = sc.c_[0] + sc.r_ * np.outer(np.sin(phi), np.cos(theta))
    y = sc.c_[1] + sc.r_ * np.outer(np.sin(phi), np.sin(theta))
    z = sc.c_[2] + sc.r_ * np.outer(np.cos(phi), np.ones_like(theta))
    axes.plot_surface(x, y, z, color='white', edgecolor='grey', alpha=0.3)
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    axes.legend(loc='upper center')
    plt.title("Spherical Classification")
    plt.show()