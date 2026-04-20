#PROGETTO MACHINE LEARNING - Nadia Masala
#Main per CLASSIFICAZIONE SFERICO - 3D

import math
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from circle_classifier import Circle_Classifier

matplotlib.use('TkAgg')

# Creazione dataset casuale
X, y = make_blobs(n_samples=200, centers=2, n_features=3, cluster_std=0.5)
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

# Definizione dei possibili parametri e selezione dei migliori
C1_par = [2, 5, 10, 15, 20]
C2_par = [2, 5, 10, 15, 20]
center_par = ['fixed','free']
selected_parameters = {'C1':C1_par, 'C2':C2_par, 'center':center_par}
cc_grid = GridSearchCV(Circle_Classifier(), selected_parameters, cv=5, verbose = 10, n_jobs = 5)
cc_grid.fit(X_train, y_train)
best_params = cc_grid.best_params_
print(best_params)

# Classificazione sferica
cc = Circle_Classifier(C1 = best_params['C1'], C2 = best_params['C2'], center = best_params['center'])
cc.fit(X_train, y_train)
y_pred = cc.predict(X_train)
print('accuracy_train =' + str(accuracy_score(y_train, y_pred)))
y_pred = cc.predict(X_test)
print('accuracy_test =' + str(accuracy_score(y_test, y_pred)))

# Grafico 3D dei punti
figure = plt.figure()
axes = figure.add_subplot(111, projection='3d')
axes.scatter(X_neg[:,0], X_neg[:,1], X_neg[:,2], facecolor = "none", edgecolor = "b", s = 50)
axes.scatter(X_pos[:,0], X_pos[:,1], X_pos[:,2], facecolor = "none", edgecolor = "r", s = 50)

# Parametrizzazione sfera in coordinate sferiche
theta = np.linspace(0, 2*np.pi, 20)
phi = np.linspace(0, np.pi, 20)
x = cc.c_[0] + cc.r_ * np.outer(np.sin(phi),np.cos(theta))
y = cc.c_[1] + cc.r_ * np.outer(np.sin(phi),np.sin(theta))
z = cc.c_[2] + cc.r_ * np.outer(np.cos(phi),np.ones_like(theta))

# Grafico 3D dela sfera
axes.plot_surface(x,y,z,color='white', edgecolor='lightblue', alpha=0.3)
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_zlabel('z')
plt.title("Classificazione Sferica")
plt.show()