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
    n_regions = 0  # number of dense regions
    start = 0  # starting index
    n_iters = 0  # number of iterations
    dense = 0  # flag is 1 if the last point is added to a dense region, 0 otherwise
    regions = []  # list of dense and no dense (empty) regions
    outliers = []  # list of outliers
    while start+l <= len(x):
        n_iters += 1
        window = x[start:start+l]

        print(window)

        distances = []  # list of distances between all couples of points in the current window
        for j in range(len(window)):
            for k in range(len(window)):
                if k < j:
                    distances.append(np.linalg.norm(window[j] - window[k]))
        d_max = max(distances)
        if d_max <= d:
            if dense == 0:  # if we are at the first iteration or the previous region is not dense
                # define a new dense region with the points of the current window
                regions.append(window)
                n_regions += 1
                start += 1  # slide
                dense = 1
            elif dense == 1:  # if at the previous iteration the point was added to a dense region
                # add the last point of the current window in the last dense region found
                regions[-1].append(window[-1])
                start += 1  # slide
        elif d_max > d:
            if dense == 0:  # if we are at the first iteration or the previous region is not dense
                outliers.extend(window[:-1])
            elif dense == 1:  # if at the previous iteration we had a dense region
                dense = 0
            if start+l-1 > len(x)-l:
                start += 1
            else:
                start = start+l-1


    return n_regions, regions, outliers, n_iters







