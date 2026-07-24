# Helper functions for Spherical Clustering

import numpy as np
from New_Spherical_Class_class import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Sliding Window Algorithm for detection of density regions along the real line
def sliding_window(x,l,d):
    n_regions = 0  # number of dense regions
    start = 0  # starting index
    n_iters = 0  # number of iterations
    dense = 0  # flag: 1 if the last point is added to a dense region, 0 otherwise
    regions = []  # list of dense regions of points
    outliers = []  # list of outliers
    while start+l <= len(x):
        n_iters += 1
        window = x[start:start+l]

        print(window)

        distances = []  # list of distances between all couples of points in the current window

        for j in range(len(window)):
            while j+1 <= len(window)-1:
                distances.append(np.linalg.norm(window[j+1] - window[j]))
                break

        d_max = max(distances)
        d_max = round(d_max,5)  # to handle numerical errors due to floating-point representation

        print(d_max)

        if d_max <= d:
            if dense == 0:  # if we are at the first iteration or the previous window was not a dense region
                # define a new dense region with the points of the current window
                regions.append(window)
                n_regions += 1
                start += 1  # slide
                dense = 1
            elif dense == 1:  # if at the previous iteration we had a dense region
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


def spherical_clustering_fit(X,l,d,eps):
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

    # elimination of (possible) multiple points in X_pca
    X_pca_list = []
    X_pca_list_idx = []
    X_pca_multi_idx = []  # list of indexes of (possible) multiple points
    for i in range(m):
        if X_pca[i] not in X_pca_list:
            X_pca_list.append(X_pca[i])
            X_pca_list_idx.append(i)
        else:
            X_pca_multi_idx.append(i)

    X_pca_sorted = X_pca_list.sorted()

    n_regions, regions, outliers, n_iter = sliding_window(X_pca_sorted,l,d)

    # clusters of points in R^1
    regions_idx = [[] for _ in range(n_regions)]
    outliers_idx = []
    for idx in X_pca_list_idx:
        xp = X_pca[idx]
        for r_idx,reg in zip(regions_idx,regions):
            if xp in reg:
                r_idx.append(idx)
        if xp in outliers:
            outliers_idx.append(idx)

    if len(X_pca_multi_idx) > 0:
        for idx in X_pca_multi_idx:
            xm = X_pca[idx]
            for r_idx,reg in zip(regions_idx,regions):
                if xm in reg:
                    r_idx.append(idx)
            if xm in outliers:
                outliers_idx.append(idx)

    # labels for points in R^n
    y_pca = np.zeros(m)
    labels = range(n_regions+1)
    for r_idx,l in zip(regions_idx,labels[1:]):
        for i in r_idx:
            y_pca[i] = l
    #for j in outliers_idx:
    #    y_pca[j] = labels[0]

    # Multiclass Spherical Classification - 1-vs-all
    y_temp = np.copy(y_pca)
    r_stack = []
    c_stack = []
    for l in labels[1:]:
        # creation of artificial binary dataset
        X_l = []  # box of points of X with label l for binary classification
        C_l = np.zeros(n)  # centroid of points with label l due to PCA
        for i in range(m):
            if y_pca[i] == l:
                X_l.append(X[i])
                y_temp[i] = -1
            else:
                y_temp[i] = +1
        for i in range(X_l.shape[0]):
            for j in range(n):
                C_l[j] = np.mean(X_l[:,j])
        distances_l = {}
        for i in range(X_l.shape[0]):
            distances_l[i] = np.linalg.norm(C_l - X_l[i])
        d_l_max = max(distances_l.value())
        d_max = d_l_max + eps
        for i in range(m):
            if X[i] not in X_l:
                in_box = 0  # counter for points in the box
                for j in range(n):
                    if X[i,j] >= C_l[j]-d_max and X[i,j] <= C_l[j]+d_max:
                        in_box += 1
                if in_box == n:  # X[i] is inside the box
                    X_l.append(X[i])  # we add X[i] to the class with label l
                    y_temp[i] = -1  # we assign temporary label -1

        # Binary Spherical Classification
        X_train, X_test, y_train, y_test = train_test_split(X, y_temp, test_size=0.2)
        C1_par = list(np.linspace(1e-1, 1e+4, 4))
        C2_par = list(np.linspace(1e-1, 1e+4, 4))
        center_par = ['fixed', 'free']
        selected_parameters = {'C1': C1_par, 'C2': C2_par, 'center': center_par}
        sc_grid = GridSearchCV(New_Spherical_Classifier(), selected_parameters, cv=5, verbose=10, n_jobs=10)
        sc_grid.fit(X_train, y_train)
        sc = New_Spherical_Classifier(C1=best_params['C1'], C2=best_params['C2'], center=best_params['center'])
        sc.fit(X_train, y_train)
        y_train_pred = sc.predict(X_train)
        print('Classification report - Training set - label = ',l)
        print(classification_report(y_train, y_train_pred))
        y_test_pred = sc.predict(X_test)
        print('Classification report - Test set - label = ',l)
        print(classification_report(y_test, y_test_pred))
        c_stack.append(sc.c_)
        r_stack.append(sc.r_)

    return labels, r_stack, c_stack, n_regions, regions_idx, outliers_idx, n_iter

def spherical_clust_assign_labels(X,labels,r_stack,c_stack):
    y = []
    for r,c,l in zip(r_stack,c_stack,labels[1:]):
        for i in range(X.shape[0]):
            if np.linalg.norm(X[i],c) <= r:
                y[i] = l
    return y














