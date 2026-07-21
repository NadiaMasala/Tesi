import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Helper_Spherical_Clustering import *

#points = [0,0.06,0.12,0.25,0.5,1,1.62,1.75,1.87,1.93,2]  # (d = 0.2)
#points = [-0.9,-0.7,-0.3,-0.1,0,0.2,0.35,0.4,0.9,1]  # (d = 0.3)
#points = [-4.8,-4.7,-4.62,-4.55,-1,-0.2,1.5,1.62,1.78,4.9]  # (d = 0.3)
points = [0,0.06,0.12,0.12,0.25,0.5,0.5,1,1.62,1.75,1.87,1.87,1.93,2]  # (d = 0.2) with double points
#points = [-4.8,-4.7,-4.62,-4.62,-4.55,-1,-1,-1,-0.2,1.5,1.62,1.62,1.62,1.78,4.9]  # (d = 0.3) with double and triple points
l = 3
d = 0.2
#d = 0.3

#points_dict = {}
#for i in range(len(points)):
#    points_dict[i] = points[i]

# elimination of (possible) double points
points_list = []
points_list_idx = []
double_points_idx = []  # list of indexes of double points
for i in range(len(points)):
    if points[i] not in points_list:
        points_list.append(points[i])
        points_list_idx.append(i)
    else:
        double_points_idx.append(i)

n_regions, regions, outliers, n_iters = sliding_window_alg(points_list,l,d)

# classification of points
regions_idx = [[] for _ in range(n_regions)]
outliers_idx = []
for idx in points_list_idx:
    p = points[idx]
    for r_idx,reg in zip(regions_idx,regions):
        if p in reg:
            r_idx.append(idx)
    if p in outliers:
        outliers_idx.append(idx)

# classification of double points
if len(double_points_idx) > 0:
    for idx in double_points_idx:
        #pd = points_dict[idx]
        pd = points[idx]
        for r_idx,reg in zip(regions_idx,regions):
            if pd in reg:
                r_idx.append(idx)
        if pd in outliers:
            outliers_idx.append(idx)

print('\n')
print('n_regions = '+str(n_regions)+'\n')
print('regions = '+str(regions)+'\n')
print('regions_idx = '+str(regions_idx)+'\n')
print('outliers = '+str(outliers)+'\n')
print('outliers_idx = '+str(outliers_idx)+'\n')
print('n_iters = '+str(n_iters)+'\n')

figure, axes = plt.subplots()
colors = cm.rainbow(np.linspace(0,1,n_regions))
for reg, c in zip(regions,colors):
    axes.scatter(reg,np.zeros(len(reg)), facecolor=c, edgecolor=c)
axes.scatter(outliers, np.zeros(len(outliers)), facecolor='gray', edgecolor='gray')
axes.set_xticks(points)
plt.show()