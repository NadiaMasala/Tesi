# Main for examples of Spherical Classification

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
from sklearn.metrics import accuracy_score, f1_score

ns = 50
nf = 2

with open('experiments/example_fixed_'+str(ns)+'_'+str(nf)+'.txt', 'w') as f:

    X, y = make_blobs(n_samples=50, centers=2, center_box=(-5,5), n_features=2, cluster_std=1.4, random_state=42)

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Splitting training classes by their labels
    A_train = []
    B_train = []
    for i in range(X_train.shape[0]):
        if y_train[i] == labels[0]:
            A_train.append(X_train[i])
        elif y_train[i] == labels[1]:
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

    # Selection of values of hyperparameters by Grid Search
    epsilon_par = list(np.linspace(d_min,d_max,5))
    minpts_par = [5, 10, 15]
    C1_par = list(np.linspace(1e-1, 1e+4, 4))
    C2_par = list(np.linspace(1e-1, 1e+4, 4))
    #center_par = ['fixed','free']
    center_par = 'fixed'
    #selected_parameters = {'epsilon':epsilon_par, 'minpts':minpts_par, 'C1':C1_par, 'C2':C2_par, 'center':center_par}
    selected_parameters = {'epsilon':epsilon_par, 'minpts':minpts_par, 'C1':C1_par, 'C2':C2_par}
    sc_grid = GridSearchCV(New_Spherical_Classifier(), selected_parameters, cv=5, verbose = 10, n_jobs = 10)
    sc_grid.fit(X_train, y_train)
    best_params = sc_grid.best_params_
    f.write('Best hyperparameters = '+ str(best_params) + '\n')

    # Spherical Classification
    sc = New_Spherical_Classifier(epsilon = best_params['epsilon'], minpts = best_params['minpts'], C1 = best_params['C1'], C2 = best_params['C2'], center = center_par)
    sc.fit(X_train, y_train)
    f.write('Class in = '+ str(sc.in_label_) + '\n')
    f.write('Optimal center = '+ str(sc.c_) + '\n')
    y_train_pred = sc.predict(X_train)
    f.write('Classification report - Training set \n')
    f.write(classification_report(y_train, y_train_pred) + '\n')
    y_test_pred = sc.predict(X_test)
    f.write('Classification report - Test set \n')
    f.write(classification_report(y_test, y_test_pred) + '\n')
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    acc_tot = accuracy_score(y, sc.predict(X))
    f1_train = f1_score(y_train, y_train_pred)
    f1_test = f1_score(y_test, y_test_pred)
    f1_tot = f1_score(y, sc.predict(X))
    f.write(str(ns)+ '&' +str(nf)+ '&' +str(round(acc_train,3))+ '&' +str(round(f1_train,3))+ '&' +str(round(acc_test,3))+ '&' +str(round(f1_test,3))+ '&' +str(round(acc_tot,3))+ '&' +str(round(f1_tot,3))+ '\\\\')


    # Graphics
    if nf == 2:
        figure, axes = plt.subplots()
        a_scatter = axes.scatter(A[:, 0], A[:, 1], facecolor="none", edgecolor="b", s=50, label='A')
        b_scatter = axes.scatter(B[:, 0], B[:, 1], facecolor="none", edgecolor="r", s=50, label='B')
        circle = plt.Circle((sc.c_[0], sc.c_[1]), sc.r_, color='black', fill=False)
        axes.add_artist(circle)
        axes.set_aspect(1)
        all_x0 = np.concatenate((X[:, 0], [sc.c_[0] - sc.r_, sc.c_[0] + sc.r_]))
        all_x1 = np.concatenate((X[:, 1], [sc.c_[1] - sc.r_, sc.c_[1] + sc.r_]))
        axes.set_xlim(min(all_x0) - 1, max(all_x0) + 1)
        axes.set_ylim(min(all_x1) - 1, max(all_x1) + 1)
        axes.legend(handles=[a_scatter,b_scatter])
        plt.title("Spherical Classification - n_samples = "+str(ns)+", n_features = "+str(nf))
        plt.savefig('experiments/example_fixed_fig_'+str(ns)+'_'+str(nf)+'.pdf')