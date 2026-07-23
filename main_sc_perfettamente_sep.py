#PROGETTO MACHINE LEARNING - Nadia Masala
#Main per CLASSIFICAZIONE SFERICO: esempio perfettamente separabile 2D

import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import label_binarize
from Spherical_Class_class import Spherical_Classifier

#X = np.array([[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[0,0],[0,0.5],[0.5,0],[0,-0.5],[-0.5,0]])
#y = np.array([+1,+1,+1,+1,+1,+1,+1,+1,-1,-1,-1,-1,-1])
#y = np.array([-1,-1,-1,-1,-1,-1,-1,-1,+1,+1,+1,+1,+1])

ns_in = 25
ns_out = 12
ns = ns_in + 2*ns_out
nf = 2
X_in = 2*np.random.random_sample((ns_in,nf))-1
y_in = np.zeros(ns_in)
X_out = np.random.random_sample((ns_out,nf))-2
X_out = np.append(X_out, np.random.random_sample((ns_out,nf))+1, axis=0)
y_out = np.ones(2*ns_out)
X = np.append(X_in,X_out,axis=0)
y = np.append(y_in,y_out,axis=0)

m = X.shape[0]
n = X.shape[1]

center = np.zeros(n)

distances_in = {}
for i in range(ns_in):
    distances_in[i] = np.linalg.norm(center - X_in[i])

radius = max(distances_in.values())

figure, axes = plt.subplots()
in_scatter = axes.scatter(X_in[:,0], X_in[:,1], facecolor = "none", edgecolor = "b", s = 50, label='A')
out_scatter = axes.scatter(X_out[:,0], X_out[:,1], facecolor = "none", edgecolor = "r", s = 50, label='B')
circle = plt.Circle(center, radius, color='black', fill=False)
axes.add_artist(circle)
axes.set_aspect(1)
all_x0 = np.concatenate((X[:,0],[-radius, radius]))
all_x1 = np.concatenate((X[:,1],[-radius, radius]))
axes.set_xlim(min(all_x0)-1,max(all_x0)+1)
axes.set_ylim(min(all_x1)-1,max(all_x1)+1)
axes.legend(handles=[in_scatter,out_scatter])
plt.savefig('experiments/perf_sep_fixed_fig_'+str(ns)+'_'+str(nf)+'.pdf')

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





