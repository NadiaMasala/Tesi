import numpy as np
from Helper_Spherical_Clustering import *

points = [0,0.0625,0.125,0.25,0.5,1,1.625,1.75,1.875,1.9375,2]
l = 3
d = 0.2

n_regions, regions, outliers, n_iters = sliding_window(points,l,d)
print('n_regions = '+str(n_regions)+'\n')
print('regions = '+str(regions)+'\n')
print('outliers = '+str(outliers)+'\n')
print('n_iters = '+str(n_iters)+'\n')