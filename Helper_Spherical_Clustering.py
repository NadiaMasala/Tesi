# Helper functions for Spherical Clustering

import numpy as np
from New_Spherical_Class_class import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def spherical_clustering_fit(X,l,d):
    m = X.shape[0]
    n = X.shape[1]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    pca = PCA(n_components = 1,copy=True)
    X_pca = pca.fit_transform(X)

    X_pca_dict = {}
    for i in range(m):
        X_pca_dict[i] = X_pca[i]

    # elimination of double points in X_pca
    X_pca_list = []
    X_pca_double_ind = []
    for i in range(m):
        if X_pca_list == []:
            X_pca_list.append(X_pca[i])
        elif X_pca[i] not in X_pca_list:
            X_pca_list.append(X_pca[i])
        else:
            X_pca_double_ind.append(i)

    X_pca_sorted = X_pca_list.sort()



# Sliding Window Algorithm for density regions along the real line
def sliding_window(x,l,d):

    for i in range(x.shape[0]):
        window = x[i:i+l]
        distances = []
        for j in range(window.shape[0]):
            for k in range(window.shape[0]):
                if k!=j:
                    distances.append(np.linalg.norm(window[j] - window[k]))
        d_max = max(distances)
        if d_max <= d:


