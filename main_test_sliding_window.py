import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Helper_Spherical_Clustering import *

points = [0,0.06,0.12,0.25,0.5,1,1.62,1.75,1.87,1.93,2]  # (d = 0.2)
#points = [-0.9,-0.7,-0.3,-0.1,0,0.2,0.35,0.4,0.9,1]  # (d = 0.3)
#points = [-4.8, -4.7, -4.62, -4.55, -1, -0.2, 1.5, 1.62, 1.78, 4.9]  # (d = 0.3)
l = 3
d = 0.2
#d = 0.3

n_regions, regions, outliers, n_iters = sliding_window(points,l,d)
print('\n')
print('n_regions = '+str(n_regions)+'\n')
print('regions = '+str(regions)+'\n')
print('outliers = '+str(outliers)+'\n')
print('n_iters = '+str(n_iters)+'\n')

figure, axes = plt.subplots()
colors = cm.rainbow(np.linspace(0,1,n_regions))
for reg, c in zip(regions,colors):
    axes.scatter(reg,np.zeros(len(reg)), facecolor=c, edgecolor=c)
axes.scatter(outliers, np.zeros(len(outliers)), facecolor='gray', edgecolor='gray')
axes.set_xticks(points)
plt.show()