import numpy as np
from sklearn.datasets import make_blobs
from K_means_class import *
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.cluster import DBSCAN
from Spectral_Clust_class import *

X, y = make_blobs(n_samples=300,centers=3,random_state=42,n_features=2)

spect_clust = Spectral_Clustering(n_clusters=3, max_iter=100, similarity='euclide')
spect_clust.fit(X)
labels = spect_clust.assign_labels(X)

DB_index = davies_bouldin_score(X, labels)
print(DB_index)
SC_index = silhouette_score(X, labels)
print(SC_index)
#i valori sono brutti quando la funzione di similarità usata (coseno) è brutta, può anche dare valori negativi



