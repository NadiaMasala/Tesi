# Main for Spherical Classification  with selection of class in the optimal sphere
# Synthetic datasets

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

n_samples = [100, 200, 300]
n_features = [2, 10, 40]

for i in n_samples:
    for j in n_features:
        # Creation of a casual dataset with 2 clusters
        X, y = make_blobs(n_samples=i, centers=2, n_features=j, cluster_std=0.6)

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

        # Splitting the dataset in training set e test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Definition of the barycenter of training points
        barycenter = np.zeros(X_train.shape[1])
        for j in range(X_train.shape[1]):
            barycenter[j] = np.mean(X_train[:, j])

        distances = {}
        for i in range(X_train.shape[0]):
            distances[i] = np.linalg.norm(barycenter - X_train[i])

        d_min = min(distances.values())
        d_max = max(distances.values())

        # Selection of values of hyperparameters by Grid Search
        epsilon_par = list(np.linspace(d_min,d_max,5))
        minpts_par = [3, 5, 10, 15, 20]
        C1_par = list(np.linspace(1e0, 1e+4, 5))
        C2_par = list(np.linspace(1e0, 1e+4, 5))
        center_par = ['fixed','free']
        selected_parameters = {'epsilon':epsilon_par, 'minpts':minpts_par, 'C1':C1_par, 'C2':C2_par, 'center':center_par}
        sc_grid = GridSearchCV(New_Spherical_Classifier(), selected_parameters, cv=5, verbose = 10, n_jobs = 10)
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