# Helper functions for Spherical Clustering

import numpy as np
from New_Spherical_Class_class import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

        # alternatively, but it is too slow
        # distance between consecutive points of the window
        #for j in range(len(window)):
        #    while j+1 <= len(window)-1:
        #        distances.append(np.linalg.norm(window[j+1] - window[j]))

        d_max = max(distances)
        d_max = round(d_max,5)  # to handle numerical errors due to floating-point representation

        print(d_max)

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
            if dense == 0:  # if we are at the first iteration or at the previous iteration we had a non-dense region
                if start+l-1 == len(x)-1:  # if we are at the last iteration
                    if window[0] in outliers:
                        outliers.extend(window[1:])  # all the points of the current window, but the first one, are outliers
                    else:
                        outliers.extend(window)  # all the points of the current window are outliers
                else:
                    outliers.extend(window[:-1])  # all the points of the current window are outliers except for the last one, we will examine it in the following iteration
            elif dense == 1:  # if at the previous iteration we had a dense region
                if start+l-1 == len(x)-1:  # if we are at the last iteration
                    outliers.append(window[-1])  # the last point is an outlier
                dense = 0
            if start+l-1 > len(x)-l:  # in order to study all the points
                start += 1
            else:
                start = start+l-1


    return n_regions, regions, outliers, n_iters

def spherical_clustering_fit(X,l,d):
    m = X.shape[0]
    n = X.shape[1]

    # scaling for PCA
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # reduction to dimension 1
    pca = PCA(n_components = 1,copy=True)
    X_pca = pca.fit_transform(X)

    # saving of indexes of points in a dictionary
    #X_pca_dict = {}
    #for i in range(m):
    #    X_pca_dict[i] = X_pca[i]

    # elimination of (possible) double points in X_pca
    X_pca_list = []
    X_pca_list_idx = []
    X_pca_double_idx = []  # list of indexes of double points
    for i in range(m):
        if X_pca[i] not in X_pca_list:
            X_pca_list.append(X_pca[i])
            X_pca_list_idx.append(i)
        else:
            X_pca_double_idx.append(i)

    X_pca_sorted = X_pca_list.sorted()

    n_regions, regions, outliers, n_iter = sliding_window(X_pca_sorted,l,d)

    # classification of points
    regions_idx = [[] for _ in range(n_regions)]
    outliers_idx = []
    for idx in X_pca_list_idx:
        xp = X_pca[idx]
        for r_idx,reg in zip(regions_idx,regions):
            if xp in reg:
                r_idx.append(idx)
        if xp in outliers:
            outliers_idx.append(idx)

    # classification of (possible) double points
    if len(X_pca_double_idx) > 0:
        for idx in X_pca_double_idx:
            xd = X_pca[idx]
            for r_idx,reg in zip(regions_idx,regions):
                if xd in reg:
                    r_idx.append(idx)
            if xd in outliers:
                outliers_idx.append(idx)



    return n_regions, regions, regions_idx, outliers, outliers_idx, n_iter











