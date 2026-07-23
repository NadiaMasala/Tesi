import numpy as np
from Helper_Spherical_Clustering import *

class Spherical_Clustering:

    def __init__(self,l=3,d=0.3,C1=1.0,C2=1.0,center='free',eps=1.0):
        self.l = l
        self.d = d
        self.C1 = C1
        self.C2 = C2
        self.center = center
        self.eps = eps

    def fit(self,X):
        self.X_ = X

        r_stack, c_stack, n_regions, regions_idx, outliers_idx, n_iter = spherical_clustering_fit(self.X_,self.l,self.d,self.C1,self.C2,self.center,self.eps)
        self.r_stack = r_stack
        self.c_stack = c_stack
        self.n_regions = n_regions
        self.regions_idx = regions_idx
        self.outliers_idx = outliers_idx
        self.n_iter = n_iter
        
        return self