#Main graphics for experiments - fixing n_samples or n_features

import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from Spherical_Class_class import Spherical_Classifier
from sklearn.svm import SVC


#n_samples = [150]*7 + [200]*7 + [300]*8
n_features = [2]*3 + [5]*3 + [10] + [2]*3 + [5]*2 + [10]*2 + [2]*3 + [5]*3 + [10]*2


#DB =[0.238,0.913,0.404,0.474,0.922,1.668,0.289,0.411,0.675,0.928,0.43,0.917,0.207,1.066,0.36,0.468,0.834,0.692,1.27,0.778,0.233,1.633]
#SC = [0.818,0.569,0.683,0.669,0.515,0.221,0.798,0.707,0.555,0.372,0.688,0.498,0.855,0.323,0.731,0.576,0.51,0.503,0.391,0.288,0.837,0.368]
best_score = [0.866,0.813,0.798,0.829,0.717,0.569,0.846,0.787,0.757,0.651,0.782,0.78,0.88,0.82,0.803,0.834,0.687,0.879,0.623,0.618,0.872,0.71]

plt.figure()
#plt.plot(range(len(n_samples)), DB, marker='o', color='red')
#plt.plot(range(len(n_samples)), SC, marker='o', color='blue')
#plt.plot(range(len(n_samples)), best_score, marker='o', color='green')
#plt.plot(range(len(n_features)), DB, marker='o', color='red')
#plt.plot(range(len(n_features)), SC, marker='o', color='blue')
plt.plot(range(len(n_features)), best_score, marker='o', color='green')
#plt.xticks(range(len(n_samples)),n_samples,rotation=45)
plt.xticks(range(len(n_features)),n_features,rotation=45)
plt.ylim(0, 1)
#plt.xlabel('n. samples')
plt.xlabel('n. features')
#plt.ylabel('DB')
#plt.ylabel('SC')
plt.ylabel('best score')
#plt.savefig('clustering_experiments/samples_bs.pdf')
plt.savefig('clustering_experiments/features_bs.pdf')