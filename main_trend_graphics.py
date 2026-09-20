#Main graphics for experiments - fixing n_samples or n_features

import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from Spherical_Class_class import Spherical_Classifier
from sklearn.svm import SVC
''''
from main_rev import n_features
'''
#print('decr2005 = '+str(round(0.755-0.698,3))+'\n')

DBSC_avg_SC = [0.784,0.754,0.702,0.804,0.635,0.596,0.777,0.706,0.783,0.755,0.692,0.822,0.722,0.668,0.774,0.698,0.779,0.746,0.805,0.672,0.777,0.725]
DBSC_avg_kM = [0.872,0.735,0.78,0.817,0.801,0.8,0.824,0.815,0.872,0.761,0.7,0.821,0.802,0.798,0.822,0.813,0.874,0.75,0.819,0.798,0.823,0.812]

DBSC_avg_SC2 = [0.784,0.754,0.702,0.804,0.783,0.755,0.692,0.822,0.722,0.668,0.779,0.746,0.805,0.672]
DBSC_avg_DBSCAN = [0.99,0.949,0.974,0.952,0.978,0.971,0.932,0.956,0.949,0.977,0.888,0.993,0.972,0.929]

diff_DBSC_avg_kM = np.abs(np.array(DBSC_avg_SC)-np.array(DBSC_avg_kM))
diff_DBSC_avg_DBSCAN = np.abs(np.array(DBSC_avg_SC2)-np.array(DBSC_avg_DBSCAN))
diff_DBSC_avg_kM_avg = np.mean(diff_DBSC_avg_kM)
diff_DBSC_avg_DBSCAN_avg = np.mean(diff_DBSC_avg_DBSCAN)
print('avg DBSC_avg difference (sc-km) = '+str(round(diff_DBSC_avg_kM_avg,3)))
print('avg DBSC_avg difference (sc-dbscan) = '+str(round(diff_DBSC_avg_DBSCAN_avg,3)))
quit()

'''
sc_DB1 = [0.412,0.424,0.651,0.357,0.816,1.001,0.42,0.657,0.41,0.421,0.686,0.32,0.529,0.743,0.43,0.67,0.424,0.454,0.357,0.748,0.424,0.587]
km_DB = [0.216,0.551,0.394,0.334,0.369,0.37,0.317,0.336,0.22,0.457,0.677,0.324,0.366,0.377,0.322,0.342,0.212,0.493,0.329,0.376,0.32,0.344]

sc_DB2 = [0.412,0.424,0.651,0.357,0.41,0.421,0.686,0.32,0.529,0.743,0.424,0.454,0.357,0.748]
dbscan_DB = [0.014,0.075,0.037,0.071,0.032,0.042,0.103,0.065,0.075,0.033,0.187,0.009,0.041,0.11]

sc_SC1 = [0.72,0.612,0.595,0.742,0.439,0.384,0.698,0.617,0.715,0.613,0.582,0.772,0.582,0.524,0.697,0.593,0.71,0.608,0.746,0.543,0.705,0.638]
km_SC = [0.845,0.649,0.687,0.767,0.741,0.74,0.778,0.763,0.847,0.672,0.606,0.774,0.743,0.74,0.775,0.761,0.848,0.659,0.772,0.737,0.778,0.761]

sc_SC2 = [0.72,0.612,0.595,0.742,0.715,0.613,0.582,0.772,0.582,0.524,0.71,0.608,0.746,0.543]
dbscan_SC = [0.988,0.936,0.968,0.94,0.976,0.966,0.914,0.948,0.938,0.973,0.866,0.992,0.965,0.912]

diff_DB_km = np.abs(np.array(sc_DB1)-np.array(km_DB))
diff_DB_dbscan = np.abs(np.array(sc_DB2)-np.array(dbscan_DB))
diff_SC_km = np.abs(np.array(sc_SC1)-np.array(km_SC))
diff_SC_dbscan = np.abs(np.array(sc_SC2)-np.array(dbscan_SC))

max_diffDB_km = np.max(diff_DB_km)
max_diffDB_dbscan = np.max(diff_DB_dbscan)
max_diffSC_km = np.max(diff_SC_km)
max_diffSC_dbscan = np.max(diff_SC_dbscan)

diff_DB_km_avg = np.mean(diff_DB_km)
diff_DB_dbscan_avg = np.mean(diff_DB_dbscan)
diff_SC_km_avg = np.mean(diff_SC_km)
diff_SC_dbscan_avg = np.mean(diff_SC_dbscan)

print('max DB difference (sc-km) = '+str(round(max_diffDB_km,3)))
print('max DB difference (sc-dbscan) = '+str(round(max_diffDB_dbscan,3)))
print('max SC difference (sc-km) = '+str(round(max_diffSC_km,3)))
print('max SC difference (sc-dbscan) = '+str(round(max_diffSC_dbscan,3)))
print('\n')
print('avg DB difference (sc-km) = '+str(round(diff_DB_km_avg,3)))
print('avg DB difference (sc-dbscan) = '+str(round(diff_DB_dbscan_avg,3)))
print('avg SC difference (sc-km) = '+str(round(diff_SC_km_avg,3)))
print('avg SC difference (sc-dbscan) = '+str(round(diff_SC_dbscan_avg,3)))

quit()
'''
'''
#accuracy_train = [0.988,0.9,1.0,0.9,1.0,0.838,0.8,0.838,0.5,0.956,0.75,0.487]
#accuracy_test = [1.0,0.9,0.95,0.8,0.975,0.825,0.8,0.9,0.5,0.925,0.775,0.5]
f1_train = [0.987,0.889,1.0,0.901,1.0,0.86,0.75,0.806,0.0,0.957,0.798,0.655]
f1_test = [1.0,0.889,0.952,0.8,0.976,0.851,0.75,0.9,0.0,0.919,0.816,0.667]
#differences = np.abs(np.array(accuracy_train)-np.array(accuracy_test))
differences = np.abs(np.array(f1_train)-np.array(f1_test))
max_diff = np.max(differences)
print(differences)
print(round(max_diff,3))
'''
'''
n_samples = [150]*8 + [200]*8 + [300]*6
n_features = [2]*8 + [5]*8 + [10]*6

# da modificare queste misure
# chiedi se ha senso fare un grafico che mette a confronto i risultati
# per i tre algoritmi di clustering e in caso se farlo sia con in
# ascissa n_samples e sia con n_features, in caso se ha senso ordinare
# in verso crescente il numero delle features o tenere l'ordine dei
# dataset usati e che si trova nella tabella
DB_s =[0.238,0.913,0.404,0.474,0.922,1.668,0.289,0.411,0.675,0.928,0.43,0.917,0.207,1.066,0.36,0.468,0.834,0.692,1.27,0.778,0.233,1.633]
SC_s = [0.818,0.569,0.683,0.669,0.515,0.221,0.798,0.707,0.555,0.372,0.688,0.498,0.855,0.323,0.731,0.576,0.51,0.503,0.391,0.288,0.837,0.368]
best_score_s = [0.866,0.813,0.798,0.829,0.717,0.569,0.846,0.787,0.757,0.651,0.782,0.78,0.88,0.82,0.803,0.834,0.687,0.879,0.623,0.618,0.872,0.71]

DB_f =[0.238,0.913,0.404,0.474,0.922,1.668,0.289,0.411,0.675,0.928,0.43,0.917,0.207,1.066,0.36,0.468,0.834,0.692,1.27,0.778,0.233,1.633]
SC_f = [0.818,0.569,0.683,0.669,0.515,0.221,0.798,0.707,0.555,0.372,0.688,0.498,0.855,0.323,0.731,0.576,0.51,0.503,0.391,0.288,0.837,0.368]
best_score_f = [0.866,0.813,0.798,0.829,0.717,0.569,0.846,0.787,0.757,0.651,0.782,0.78,0.88,0.82,0.803,0.834,0.687,0.879,0.623,0.618,0.872,0.71]


plt.figure()
plt.plot(range(len(n_samples)), DB, marker='o', color='red', label='DB')
plt.plot(range(len(n_samples)), SC, marker='o', color='blue', label='SC')
plt.plot(range(len(n_samples)), best_score, marker='o', color='green', label='best_score')
plt.xticks(range(len(n_samples)),n_samples,rotation=45)
plt.xlabel('n. samples')
plt.legend()
plt.tight_layout()
plt.savefig('clustering_experiments/samples_clustering.pdf')

plt.figure()
plt.plot(range(len(n_features)), DB, marker='o', color='red', label='DB')
plt.plot(range(len(n_features)), SC, marker='o', color='blue', label='SC')
plt.plot(range(len(n_features)), best_score, marker='o', color='green', label='best_score')
plt.xticks(range(len(n_features)),n_features,rotation=45)
plt.xlabel('n. features')
plt.legend()
plt.tight_layout()
plt.savefig('clustering_experiments/features_clustering.pdf')
'''
'''
n_samples100 = 100
n_features = [2,10,40]
acc_mb100 = [1.0,0.9,0.95]
acc_mc100 = [0.8,0.9,0.5]
f1_mb100 = [1.0,0.889,0.952]
f1_mc100 = [0.75,0.9,0.0]

n_samples200 = 200
acc_mb200 = [0.8,0.975,0.825]
acc_mc200 = [0.925,0.775,0.5]
f1_mb200 = [0.8,0.976,0.851]
f1_mc200 = [0.919,0.816,0.667]

n_features2 = 2
n_samples = [100, 200]
acc_mbf2 = [1.0,0.8]
acc_mcf2 = [0.8,0.925]
f1_mbf2 = [1.0,0.8]
f1_mcf2 = [0.75,0.919]

n_features10 = 10
acc_mbf10 = [0.9,0.975]
acc_mcf10 = [0.9,0.775]
f1_mbf10 = [0.889,0.976]
f1_mcf10 = [0.9,0.816]

n_features40 = 40
acc_mbf40 = [0.95,0.825]
acc_mcf40 = [0.5,0.5]
f1_mbf40 = [0.952,0.851]
f1_mcf40 = [0.0,0.667]

plt.figure()
plt.plot(range(len(n_features)), acc_mb100, marker='o', color='red', label='accuracy')
plt.plot(range(len(n_features)), f1_mb100, marker='o', color='blue', label='F1-score')
plt.xticks(range(len(n_features)),n_features)
plt.xlabel('n. features')
plt.ylim(-0.1,1.1)
plt.legend()
plt.title('n. samples = '+str(n_samples100))
plt.tight_layout()
plt.savefig('experiments/exp_mb_class_samples_'+str(n_samples100)+'.pdf')

plt.figure()
plt.plot(range(len(n_features)), acc_mb200, marker='o', color='red', label='accuracy')
plt.plot(range(len(n_features)), f1_mb200, marker='o', color='blue', label='F1-score')
plt.xticks(range(len(n_features)),n_features)
plt.xlabel('n. features')
plt.ylim(-0.1,1.1)
plt.legend()
plt.title('n. samples = '+str(n_samples200))
plt.tight_layout()
plt.savefig('experiments/exp_mb_class_samples_'+str(n_samples200)+'.pdf')

plt.figure()
plt.plot(range(len(n_samples)), acc_mbf2, marker='o', color='red', label='accuracy')
plt.plot(range(len(n_samples)), f1_mbf2, marker='o', color='blue', label='F1-score')
plt.xticks(range(len(n_samples)),n_samples)
plt.xlabel('n. samples')
plt.ylim(-0.1,1.1)
plt.legend()
plt.title('n. features = '+str(n_features2))
plt.tight_layout()
plt.savefig('experiments/exp_mb_class_features_'+str(n_features2)+'.pdf')

plt.figure()
plt.plot(range(len(n_samples)), acc_mbf10, marker='o', color='red', label='accuracy')
plt.plot(range(len(n_samples)), f1_mbf10, marker='o', color='blue', label='F1-score')
plt.xticks(range(len(n_samples)),n_samples)
plt.xlabel('n. samples')
plt.ylim(-0.1,1.1)
plt.legend()
plt.title('n. features = '+str(n_features10))
plt.tight_layout()
plt.savefig('experiments/exp_mb_class_features_'+str(n_features10)+'.pdf')

plt.figure()
plt.plot(range(len(n_samples)), acc_mbf40, marker='o', color='red', label='accuracy')
plt.plot(range(len(n_samples)), f1_mbf40, marker='o', color='blue', label='F1-score')
plt.xticks(range(len(n_samples)),n_samples)
plt.xlabel('n. samples')
plt.ylim(-0.1,1.1)
plt.legend()
plt.title('n. features = '+str(n_features40))
plt.tight_layout()
plt.savefig('experiments/exp_mb_class_features_'+str(n_features40)+'.pdf')


plt.figure()
plt.plot(range(len(n_features)), acc_mc100, marker='o', color='red', label='accuracy')
plt.plot(range(len(n_features)), f1_mc100, marker='o', color='blue', label='F1-score')
plt.xticks(range(len(n_features)),n_features)
plt.xlabel('n. features')
plt.ylim(-0.1,1.1)
plt.legend()
plt.title('n. samples = '+str(n_samples100))
plt.tight_layout()
plt.savefig('experiments/exp_mc_class_samples_'+str(n_samples100)+'.pdf')

plt.figure()
plt.plot(range(len(n_features)), acc_mc200, marker='o', color='red', label='accuracy')
plt.plot(range(len(n_features)), f1_mc200, marker='o', color='blue', label='F1-score')
plt.xticks(range(len(n_features)),n_features)
plt.xlabel('n. features')
plt.ylim(-0.1,1.1)
plt.legend()
plt.title('n. samples = '+str(n_samples200))
plt.tight_layout()
plt.savefig('experiments/exp_mc_class_samples_'+str(n_samples200)+'.pdf')

plt.figure()
plt.plot(range(len(n_samples)), acc_mcf2, marker='o', color='red', label='accuracy')
plt.plot(range(len(n_samples)), f1_mcf2, marker='o', color='blue', label='F1-score')
plt.xticks(range(len(n_samples)),n_samples)
plt.xlabel('n. samples')
plt.ylim(-0.1,1.1)
plt.legend()
plt.title('n. features = '+str(n_features2))
plt.tight_layout()
plt.savefig('experiments/exp_mc_class_features_'+str(n_features2)+'.pdf')

plt.figure()
plt.plot(range(len(n_samples)), acc_mcf10, marker='o', color='red', label='accuracy')
plt.plot(range(len(n_samples)), f1_mcf10, marker='o', color='blue', label='F1-score')
plt.xticks(range(len(n_samples)),n_samples)
plt.xlabel('n. samples')
plt.ylim(-0.1,1.1)
plt.legend()
plt.title('n. features = '+str(n_features10))
plt.tight_layout()
plt.savefig('experiments/exp_mc_class_features_'+str(n_features10)+'.pdf')

plt.figure()
plt.plot(range(len(n_samples)), acc_mcf40, marker='o', color='red', label='accuracy')
plt.plot(range(len(n_samples)), f1_mcf40, marker='o', color='blue', label='F1-score')
plt.xticks(range(len(n_samples)),n_samples)
plt.xlabel('n. samples')
plt.ylim(-0.1,1.1)
plt.legend()
plt.title('n. features = '+str(n_features40))
plt.tight_layout()
plt.savefig('experiments/exp_mc_class_features_'+str(n_features40)+'.pdf')
'''
'''
DBSC_avg25 = [0.754,0.755,0.746]
n_samples = [150,200,300]
n_features25 = 2
n_centers25 = 5
plt.figure()
plt.plot(range(len(n_samples)), DBSC_avg25, marker='o', color='green')
plt.xticks(range(len(n_samples)),n_samples)
plt.xlabel('n. samples')
plt.ylabel('DBSC_avg')
plt.ylim(-0.1,1.1)
plt.title('n. features = '+str(n_features25)+', n. centers = '+str(n_centers25))
plt.tight_layout()
plt.savefig('experiments/exp_clust_samples_2_5.pdf')

DBSC_avg53 = [0.804,0.822,0.805]
n_features53 = 5
n_centers53 = 3
plt.figure()
plt.plot(range(len(n_samples)), DBSC_avg53, marker='o', color='green')
plt.xticks(range(len(n_samples)),n_samples)
plt.xlabel('n. samples')
plt.ylabel('DBSC_avg')
plt.ylim(-0.1,1.1)
plt.title('n. features = '+str(n_features53)+', n. centers = '+str(n_centers53))
plt.tight_layout()
plt.savefig('experiments/exp_clust_samples_5_3.pdf')

DBSC_avg55 = [0.635,0.722,0.672]
n_features55 = 5
n_centers55 = 5
plt.figure()
plt.plot(range(len(n_samples)), DBSC_avg55, marker='o', color='green')
plt.xticks(range(len(n_samples)),n_samples)
plt.xlabel('n. samples')
plt.ylabel('DBSC_avg')
plt.ylim(-0.1,1.1)
plt.title('n. features = '+str(n_features55)+', n. centers = '+str(n_centers55))
plt.tight_layout()
plt.savefig('experiments/exp_clust_samples_5_5.pdf')

DBSC_avg103 = [0.777,0.774,0.777]
n_features103 = 10
n_centers103 = 3
plt.figure()
plt.plot(range(len(n_samples)), DBSC_avg103, marker='o', color='green')
plt.xticks(range(len(n_samples)),n_samples)
plt.xlabel('n. samples')
plt.ylabel('DBSC_avg')
plt.ylim(-0.1,1.1)
plt.title('n. features = '+str(n_features103)+', n. centers = '+str(n_centers103))
plt.tight_layout()
plt.savefig('experiments/exp_clust_samples_10_3.pdf')

DBSC_avg105 = [0.706,0.698,0.725]
n_features105 = 10
n_centers105 = 5
plt.figure()
plt.plot(range(len(n_samples)), DBSC_avg105, marker='o', color='green')
plt.xticks(range(len(n_samples)),n_samples)
plt.xlabel('n. samples')
plt.ylabel('DBSC_avg')
plt.ylim(-0.1,1.1)
plt.title('n. features = '+str(n_features105)+', n. centers = '+str(n_centers105))
plt.tight_layout()
plt.savefig('experiments/exp_clust_samples_10_5.pdf')
'''
'''
DBSC_avg1503 = [0.784,0.804,0.777]
n_samples1503 = 150
n_features = [2,5,10]
n_centers1503 = 3
plt.figure()
plt.plot(range(len(n_features)), DBSC_avg1503, marker='o', color='green')
plt.xticks(range(len(n_features)),n_features)
plt.xlabel('n. features')
plt.ylabel('DBSC_avg')
plt.ylim(-0.1,1.1)
plt.title('n. samples = '+str(n_samples1503)+', n. centers = '+str(n_centers1503))
plt.tight_layout()
plt.savefig('experiments/exp_clust_features_150_3.pdf')

DBSC_avg1505 = [0.754,0.635,0.706]
n_samples1505 = 150
n_centers1505 = 5
plt.figure()
plt.plot(range(len(n_features)), DBSC_avg1505, marker='o', color='green')
plt.xticks(range(len(n_features)),n_features)
plt.xlabel('n. features')
plt.ylabel('DBSC_avg')
plt.ylim(-0.1,1.1)
plt.title('n. samples = '+str(n_samples1505)+', n. centers = '+str(n_centers1505))
plt.tight_layout()
plt.savefig('experiments/exp_clust_features_150_5.pdf')


DBSC_avg2003 = [0.783,0.822,0.774]
n_samples2003 = 200
n_centers2003 = 3
plt.figure()
plt.plot(range(len(n_features)), DBSC_avg2003, marker='o', color='green')
plt.xticks(range(len(n_features)),n_features)
plt.xlabel('n. features')
plt.ylabel('DBSC_avg')
plt.ylim(-0.1,1.1)
plt.title('n. samples = '+str(n_samples2003)+', n. centers = '+str(n_centers2003))
plt.tight_layout()
plt.savefig('experiments/exp_clust_features_200_3.pdf')


DBSC_avg3003 = [0.779,0.805,0.777]
n_samples3003 = 300
n_centers3003 = 3
plt.figure()
plt.plot(range(len(n_features)), DBSC_avg3003, marker='o', color='green')
plt.xticks(range(len(n_features)),n_features)
plt.xlabel('n. features')
plt.ylabel('DBSC_avg')
plt.ylim(-0.1,1.1)
plt.title('n. samples = '+str(n_samples3003)+', n. centers = '+str(n_centers3003))
plt.tight_layout()
plt.savefig('experiments/exp_clust_features_300_3.pdf')

DBSC_avg3005 = [0.746,0.672,0.725]
n_samples3005 = 300
n_centers3005 = 5
plt.figure()
plt.plot(range(len(n_features)), DBSC_avg3005, marker='o', color='green')
plt.xticks(range(len(n_features)),n_features)
plt.xlabel('n. features')
plt.ylabel('DBSC_avg')
plt.ylim(-0.1,1.1)
plt.title('n. samples = '+str(n_samples3005)+', n. centers = '+str(n_centers3005))
plt.tight_layout()
plt.savefig('experiments/exp_clust_features_300_5.pdf')
'''
'''
DBSC_avg = [0.99,0.949,0.974,0.952,0.978,0.971,0.932,0.956,0.949,0.977,0.888,0.993,0.993,0.972,0.929,0.948]
max = max(DBSC_avg)
min = min(DBSC_avg)
print('max = ',round(max,3))
print('min = ',round(min,3))

DBSC_avg23 = [0.99,0.978,0.888]
n_samples = [150,200,300]
n_features23 = 2
n_centers23 = 3
plt.figure()
plt.plot(range(len(n_samples)), DBSC_avg23, marker='o', color='green')
plt.xticks(range(len(n_samples)),n_samples)
plt.xlabel('n. samples')
plt.ylabel('DBSC_avg')
plt.ylim(-0.1,1.1)
plt.title('n. features = '+str(n_features23)+', n. centers = '+str(n_centers23))
plt.tight_layout()
plt.savefig('experiments/exp_clust_samples_2_3.pdf')

DBSC_avg25 = [0.949,0.971,0.993]
n_features25 = 2
n_centers25 = 5
plt.figure()
plt.plot(range(len(n_samples)), DBSC_avg25, marker='o', color='green')
plt.xticks(range(len(n_samples)),n_samples)
plt.xlabel('n. samples')
plt.ylabel('DBSC_avg')
plt.ylim(-0.1,1.1)
plt.title('n. features = '+str(n_features25)+', n. centers = '+str(n_centers25))
plt.tight_layout()
plt.savefig('experiments/exp_clust_samples_2_5.pdf')

DBSC_avg27 = [0.974,0.932,0.993]
n_features27 = 2
n_centers27 = 7
plt.figure()
plt.plot(range(len(n_samples)), DBSC_avg27, marker='o', color='green')
plt.xticks(range(len(n_samples)),n_samples)
plt.xlabel('n. samples')
plt.ylabel('DBSC_avg')
plt.ylim(-0.1,1.1)
plt.title('n. features = '+str(n_features27)+', n. centers = '+str(n_centers27))
plt.tight_layout()
plt.savefig('experiments/exp_clust_samples_2_7.pdf')

DBSC_avg53 = [0.952,0.956,0.972]
n_features53 = 5
n_centers53 = 3
plt.figure()
plt.plot(range(len(n_samples)), DBSC_avg53, marker='o', color='green')
plt.xticks(range(len(n_samples)),n_samples)
plt.xlabel('n. samples')
plt.ylabel('DBSC_avg')
plt.ylim(-0.1,1.1)
plt.title('n. features = '+str(n_features53)+', n. centers = '+str(n_centers53))
plt.tight_layout()
plt.savefig('experiments/exp_clust_samples_5_3.pdf')

print('decr23 = '+str(round(0.99-0.888,3))+'\n')
print('incr25 = '+str(round(0.993-0.949,3))+'\n')
print('incr53 = '+str(round(0.972-0.952,3))+'\n')
'''
print('decr2005 = '+str(round(0.755-0.698,3))+'\n')

'''
DBSC_avg2005 = [0.755,0.722,0.698]
n_samples2005 = 200
n_centers2005 = 5
n_features = [2,5,10]
plt.figure()
plt.plot(range(len(n_features)), DBSC_avg2005, marker='o', color='green')
plt.xticks(range(len(n_features)),n_features)
plt.xlabel('n. features')
plt.ylabel('DBSC_avg')
plt.ylim(-0.1,1.1)
plt.title('n. samples = '+str(n_samples2005)+', n. centers = '+str(n_centers2005))
plt.tight_layout()
plt.savefig('experiments/exp_clust_features_200_5.pdf')
'''
