#PROGETTO MACHINE LEARNING - Nadia Masala
#Main per CLASSIFICAZIONE SFERICO: esempio perfettamente separabile 2D

import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import label_binarize
from Spherical_Class_class import Spherical_Classifier

X = np.array([[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[0,0],[0,0.5],[0.5,0],[0,-0.5],[-0.5,0]])
#y = np.array([+1,+1,+1,+1,+1,+1,+1,+1,-1,-1,-1,-1,-1])
y = np.array([-1,-1,-1,-1,-1,-1,-1,-1,+1,+1,+1,+1,+1])

m = X.shape[0]
n = X.shape[1]

# Definition of the centroid of all points
centroid = np.zeros(n)
for j in range(n):
    centroid[j] = np.mean(X[:,j])

distances = {}
for i in range(m):
    distances[i] = np.linalg.norm(centroid - X[i])

d_min = min(distances.values())
d_max = max(distances.values())
d_mean = np.mean([d_min,d_max])

#radius_par = [np.mean([d_min,d_mean]),d_mean,np.mean([d_mean,d_max])]

radius = d_mean

A_in = []
B_in = []
for i in range(m):
    if distances[i] <= radius:
        if y[i] == -1:
            A_in.append(X[i])
        elif y[i] == +1:
            B_in.append(X[i])

if len(A_in) < len(B_in):
    y = label_binarize(y, classes=[1,-1], neg_label=-1, pos_label=1)  # invert the labels

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

figure, axes = plt.subplots()
axes.scatter(X_neg[:,0], X_neg[:,1], facecolor = "none", edgecolor = "b", s = 50)
axes.scatter(X_pos[:,0], X_pos[:,1], facecolor = "none", edgecolor = "r", s = 50)
plt.show()

quit()

# Definizione dei possibili parametri e selezione dei migliori
C1_par = [2, 5, 10, 15, 20]
C2_par = [2, 5, 10, 15, 20]
center_par = ['fixed','free']
selected_parameters = {'C1':C1_par, 'C2':C2_par, 'center':center_par}
sc_grid = GridSearchCV(Spherical_Classifier(), selected_parameters, cv=5, verbose = 10, n_jobs = 5)
sc_grid.fit(X, y)
best_params = sc_grid.best_params_
print(best_params)

# Classificazione sferica
sc = Spherical_Classifier(C1 = best_params['C1'], C2 = best_params['C2'], center = best_params['center'])
sc.fit(X, y)
y_pred = sc.predict(X)
print(sc.r_)
print(classification_report(y, y_pred))

# Grafico del predittore ottimale
figure, axes = plt.subplots()
axes.scatter(X_neg[:,0], X_neg[:,1], facecolor = "none", edgecolor = "b", s = 50)
axes.scatter(X_pos[:,0], X_pos[:,1], facecolor = "none", edgecolor = "r", s = 50)
circle = plt.Circle((sc.c_[0], sc.c_[1]), sc.r_, color='black', fill=False)
axes.add_artist(circle)
axes.set_aspect(1)
all_x0 = np.concatenate((X[:,0],[sc.c_[0] - sc.r_, sc.c_[0] + sc.r_]))
all_x1 = np.concatenate((X[:,1],[sc.c_[1] - sc.r_, sc.c_[1] + sc.r_]))
axes.set_xlim(min(all_x0)-1,max(all_x0)+1)
axes.set_ylim(min(all_x1)-1,max(all_x1)+1)
plt.show()





