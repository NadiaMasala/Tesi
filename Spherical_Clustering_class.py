# Class for Spherical Clustering

import numpy as np
from sklearn.base import BaseEstimator , ClassifierMixin
from Helper_Spherical_Clustering import *

class Spherical_Clustering( BaseEstimator , ClassifierMixin ):

    def __init__(self,l=3,d=0.3,eps=1.0):
        self.l = l
        self.d = d
        self.eps = eps

    def fit(self,X):
        labels, r_stack, c_stack, X_pca, n_regions, regions, regions_idx, outliers, outliers_idx, n_iter = spherical_clustering_fit(X,self.l,self.d,self.eps)
        self.labels = labels
        self.r_stack = r_stack
        self.c_stack = c_stack
        self.X_pca = X_pca
        self.n_regions = n_regions
        self.regions = regions
        self.regions_idx = regions_idx
        self.outliers = outliers
        self.outliers_idx = outliers_idx
        self.n_iter = n_iter

        return self

    def assign_labels(self,X):
        y, r_stack_clust, c_stack_clust = spherical_clust_assign_labels(X, self.labels, self.r_stack, self.c_stack)
        self.y = y
        self.r_stack_clust = r_stack_clust
        self.c_stack_clust = c_stack_clust

        return self.y, self.r_stack_clust, self.c_stack_clust
