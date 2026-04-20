#PROGETTO MACHINE LEARNING - Nadia Masala
#Main per CLASSIFICATORE SFERICO

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from Spherical_Class_class import Spherical_Classifier

n_features = 2

# Creazione dataset casuale
X, y = make_blobs(n_samples=200, centers=2, n_features=n_features, cluster_std=0.6)
y = label_binarize(y, classes=[-1,1], neg_label=-1, pos_label=1)

m = X.shape[0]
n = X.shape[1] 

# Suddivisione dei punti in base alla loro vera etichetta
X_neg = []
X_pos = []
for i in range(m):
    if y[i] == -1:
        X_neg.append(X[i])
    elif y[i] == +1:
        X_pos.append(X[i])
X_neg = np.array(X_neg)
X_pos = np.array(X_pos)

# Definizione dei centroidi
C_neg = np.zeros(n)
C_pos = np.zeros(n)
for j in range(n):
    C_neg[j] = np.mean(X_neg[:,j])
    C_pos[j] = np.mean(X_pos[:,j])

# Definizione di punti extra misclassificati per creare rumore nel dataset
D_neg = np.zeros(X_neg.shape[0])
D_pos = np.zeros(X_pos.shape[0])
for i in range(X_neg.shape[0]):
    D_neg[i] = np.linalg.norm(C_neg - X_neg[i])  # distanza dell'i-esimo punto negativo dal centroide negativo
R_neg = np.max(D_neg)  # raggio della zona negativa
for j in range(X_pos.shape[0]):
    D_pos[j] = np.linalg.norm(C_pos - X_pos[j])  # distanza dell'i-esimo punto positivo dal centroide positivo
R_pos = np.max(D_pos)  # raggio della zona positiva
X_extra_neg = np.zeros((math.floor(X_neg.shape[0]*(15/100)), n))  # inizializzazione di un 15% di punti extra misclassificati tra i negativi
X_extra_pos = np.zeros((math.floor(X_pos.shape[0]*(15/100)), n))  # inizializzazione di un 15% di punti extra misclassificati tra i positivi
for h in range(X_extra_neg.shape[0]):
    for k in range(X_extra_neg.shape[1]):
        X_extra_neg[h,k] = np.random.uniform(C_neg[k]-R_neg, C_neg[k]+R_neg)  # definizione random di un punto misclassificato tra i negativi
for h in range(X_extra_pos.shape[0]):
    for k in range(X_extra_pos.shape[1]):
        X_extra_pos[h,k] = np.random.uniform(C_pos[k]-R_pos, C_pos[k]+R_pos)  # definizione random di un punto misclassificato tra i positivi
y_extra_neg = np.ones((X_extra_neg.shape[0],1), dtype=np.int32)  # i nuovi punti extra nella zona negativa sono misclassificati come positivi
y_extra_pos = -np.ones((X_extra_pos.shape[0],1), dtype=np.int32)  # i nuovi punti extra nella zona positiva sono misclassificati come negativi  

# Creazione nuovo dataset con rumore
X = np.concatenate((X, X_extra_neg, X_extra_pos))
y = np.concatenate((y, y_extra_neg, y_extra_pos))
X_neg = np.concatenate((X_neg, X_extra_pos))
X_pos = np.concatenate((X_pos, X_extra_neg))

# Suddivisione del nuovo dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Definizione dei valori possibili per gli iperparametri e selezione dei migliori
C1_par = [2, 5, 10, 15, 20]
C2_par = [2, 5, 10, 15, 20]
center_par = ['fixed','free']
selected_parameters = {'C1':C1_par, 'C2':C2_par, 'center':center_par}
sc_grid = GridSearchCV(Spherical_Classifier(), selected_parameters, cv=5, verbose = 10, n_jobs = 5)
sc_grid.fit(X_train, y_train)
best_params = sc_grid.best_params_
print(best_params)

# Classificazione sferica
sc = Spherical_Classifier(C1 = best_params['C1'], C2 = best_params['C2'], center = best_params['center'])
sc.fit(X_train, y_train)
y_pred = sc.predict(X_train)
print(classification_report(y_train, y_pred))
y_pred = sc.predict(X_test)
print(classification_report(y_test, y_pred))

print(sc.c_)

if n == 2:
    # Grafico 2D dei punti
    figure, axes = plt.subplots()
    axes.scatter(X_neg[:,0], X_neg[:,1], facecolor = "none", edgecolor = "b", s = 50)
    axes.scatter(X_pos[:,0], X_pos[:,1], facecolor = "none", edgecolor = "r", s = 50)
    # Grafico della circonferenza ottimale
    circle = plt.Circle((sc.c_[0], sc.c_[1]), sc.r_, color='black', fill=False)
    axes.add_artist(circle)
    axes.set_aspect(1)
    all_x0 = np.concatenate((X[:,0],[sc.c_[0] - sc.r_, sc.c_[0] + sc.r_]))
    all_x1 = np.concatenate((X[:,1],[sc.c_[1] - sc.r_, sc.c_[1] + sc.r_]))
    axes.set_xlim(min(all_x0)-1,max(all_x0)+1)
    axes.set_ylim(min(all_x1)-1,max(all_x1)+1)
    plt.title("Classificazione Sferica")
    plt.show()
elif n == 3:
    matplotlib.use('TkAgg')  # rendo il grafico interattivo
    # Grafico 3D dei punti
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    axes.scatter(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], facecolor="none", edgecolor="b", s=50)
    axes.scatter(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], facecolor="none", edgecolor="r", s=50)
    # Parametrizzazione della sfera
    theta = np.linspace(0, 2 * np.pi, 20)
    phi = np.linspace(0, np.pi, 20)
    x = sc.c_[0] + sc.r_ * np.outer(np.sin(phi), np.cos(theta))
    y = sc.c_[1] + sc.r_ * np.outer(np.sin(phi), np.sin(theta))
    z = sc.c_[2] + sc.r_ * np.outer(np.cos(phi), np.ones_like(theta))
    # Grafico 3D della sfera ottimale
    axes.plot_surface(x, y, z, color='white', edgecolor='grey', alpha=0.3)
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    plt.title("Classificazione Sferica")
    plt.show()

