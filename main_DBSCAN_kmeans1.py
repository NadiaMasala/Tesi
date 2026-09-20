import math
from curses.ascii import DC1

import matplotlib
from itertools import product
from tqdm import tqdm
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.cluster import DBSCAN, KMeans


n_samples = [150, 200, 300]
n_features = [2,5,10]
n_centers = [3,5,7]


for m in n_samples:
    for n in n_features:
        for nc in n_centers:
            with open('clustering_experiments/new1_dataset_db_km_'+str(m)+'_'+str(n)+'_'+str(nc)+'.txt', 'w') as f:
                f.write('Synthetic dataset for clustering with n_samples=' + str(m) + ', n_features=' + str(n) + ', n_centers=' + str(nc) + ' (make_blobs - cluster_std=1.0)\n\n')

                X, y = make_blobs(n_samples=m, centers=nc, n_features=n, cluster_std=1.0,random_state=42)

                kmeans = KMeans(n_clusters=nc, random_state=42)
                kmeans.fit(X)
                y_km = kmeans.labels_

                DB_km = davies_bouldin_score(X, y_km)
                SC_km = silhouette_score(X, y_km)

                # Mapping the scores to [0,1] to compute their average
                SC_km_scaled = (SC_km + 1) / 2
                DB_km_scaled = 1 / (1 + DB_km)

                best_score_km = (SC_km_scaled + DB_km_scaled) / 2

                f.write('Results from KMeans:\n')
                f.write('Number of clusters = ' + str(nc) + '\n')
                f.write('DB_index = ' + str(DB_km) + '\n')
                f.write('SC_index = ' + str(SC_km) + '\n')
                f.write('best_score = ' + str(best_score_km) + '\n')
                f.write(' & ' + str(nc) + ' & ' + str(round(DB_km, 3)) + ' & ' + str(round(SC_km, 3)) + ' & ' + str(round(best_score_km, 3)) + '\\\\'+'\n\n')


                eps_par = [0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5]  # same as eps for boxes in spherical clustering
                minpts_par = [3, 4, 5, 7]  # same as l for sliding window algorithm

                parameter_grid = list(product(eps_par, minpts_par))

                best_score_db = 1e-2
                best_params = {'eps': 0,'minpts':0}

                for eps, minpts in tqdm(parameter_grid,desc="Grid search",unit="configuration"):
                    try:
                        dbscan = DBSCAN(eps=eps,min_samples=minpts)
                        dbscan.fit(X)
                        y_db = dbscan.labels_

                        labels = (np.unique(y_db)).tolist()

                        if labels[0] != -1:
                            n_clst_db = len(labels)
                        else:
                            n_clst_db = len(labels) - 1

                        X_no_out = []  # list of points that are not outliers
                        X_outliers_idx = []  # list of indexes of points that are outliers
                        y_db_no_out = []  # list of labels of points that are not outliers
                        for i in range(m):
                            if y_db[i] != -1:
                                X_no_out.append(X[i])
                                y_db_no_out.append(y_db[i])
                            else:
                                X_outliers_idx.append(i)

                        DB_db = davies_bouldin_score(X_no_out, y_db_no_out)
                        SC_db = silhouette_score(X_no_out, y_db_no_out)

                        # Mapping the scores to [0,1] to compute their average
                        SC_db_scaled = (SC_db + 1) / 2
                        DB_db_scaled = 1 / (1 + DB_db)

                        Avg_score = (SC_db_scaled + DB_db_scaled) / 2

                        if Avg_score > best_score_db:
                            best_params = {'eps': eps,'minpts':minpts}
                            best_score_db = Avg_score
                            DB = DB_db
                            SC = SC_db
                            n_clust = n_clst_db
                            n_out = len(X_outliers_idx)
                    except:
                        continue

                f.write('Results from DBSCAN:\n')
                f.write('Best hyperparameters = ' + str(best_params) + '\n\n')
                f.write('Number of clusters = ' + str(n_clust) + '\n')
                f.write('Number of outliers = ' + str(n_out) + '\n')
                f.write('DB_index = ' + str(DB) + '\n')
                f.write('SC_index = ' + str(SC) + '\n')
                f.write('best_score = ' + str(best_score_db) + '\n')
                f.write(' & ' + str(n_clust) + ' & ' + str(round(DB, 3)) + ' & ' + str(round(SC, 3)) + ' & ' + str(round(best_score_db, 3)) + '\\\\' + '\n\n')