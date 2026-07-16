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

    n_regions, regions, outliers, n_iter = sliding_window(X_pca_sorted,l,d)



# Sliding Window Algorithm for density regions along the real line
def sliding_window(x,l,d):
    n_regions = 0
    start = 0
    n_iters = 0
    regions = []
    outliers = []
    while start+l-1 <= len(x)-1:
        window = x[start:start+l-1]
        distances = []
        for j in range(len(window)):
            for k in range(len(window)):
                if k!=j:
                    distances.append(np.linalg.norm(window[j] - window[k]))
        d_max = max(distances)
        if d_max <= d:
            if n_iters == 0:
                regions[n_iters] = regions[n_iters].append(window)
                n_regions += 1
                start += 1
            elif len(regions[n_iters-1]) > 0:  # if the previous region is not empty (is a dense region)
                # add the last point of the current window in the previous dense region
                regions[n_iters-1] = regions[n_iters-1].append(window[-1])
                start += 1  # slide
            else: # the previous region is empty (is not dense) or does not exist
                # define a new dense region with the points of the current window
                regions[n_iters] = regions[n_iters].append(window)
                n_regions += 1
                start += 1
        elif d_max > d:
            if n_iters > 0:
                if not len(regions[n_iters-1]) > 0:
                    outliers.append(start)
            start = start+l-1
        n_iters += 1

    return n_regions, regions, outliers, n_iters







