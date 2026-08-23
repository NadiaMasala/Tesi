# Main for SVM with degree 2 polynomial kernel
# Real datasets

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from New_Spherical_Class_class import New_Spherical_Classifier
from New_Helper_SC import *
from sklearn.metrics import accuracy_score, f1_score

data_list = ['germannumer','Fertility']
list_count_0 = []
list_count_1 = []
for d in data_list:
    y = np.genfromtxt('datasets/'+d+'_label.csv',delimiter=',')
    count_0 = 0
    count_1 = 0
    for i in range(len(y)):
        if y[i] == 0:
            count_0 += 1
        elif y[i] == 1:
            count_1 += 1
    list_count_0.append(count_0)
    list_count_1.append(count_1)
print('list_count_0 = '+str(list_count_0)+'\n')
print('list_count_1 = '+str(list_count_1))
quit()
'''
acc = []
f1 = []
data_list = ['liver','blood_transfusion','flowmeters','heart','diabetes','breast','divorce','australian','Mesothelioma','Gallstone','sonar','breast_wisconsin','germannumer','Fertility','HillValley_training']

for d in data_list:
    X = np.genfromtxt('datasets/'+d+'_data.csv',delimiter=',')
    y = np.genfromtxt('datasets/'+d+'_label.csv',delimiter=',')
    X = MinMaxScaler((-1,1)).fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    m = X.shape[0]
    n = X.shape[1]

    with open('svm_experiments/'+d+'_ksvm.txt', 'w') as f:

        f.write('Dataset "'+d+'":  n_samples = '+str(m)+', n_features = '+str(n)+'\n')

        # Selection of best value of hyperparameter
        C_par = list(np.linspace(1e-1, 1e+4, 4))
        selected_parameters = {'C':C_par}
        ksvm_grid = GridSearchCV(SVC(kernel='poly', degree=2), selected_parameters, cv=5, verbose = 10, n_jobs = 10)
        ksvm_grid.fit(X_train, y_train)
        best_params = ksvm_grid.best_params_
        f.write('Best hyperparameter = '+ str(best_params) + '\n')

        # Polynomial kernel SVM
        ksvm = SVC(C=best_params['C'], kernel='poly', degree=2)
        ksvm.fit(X_train, y_train)
        y_train_pred = ksvm.predict(X_train)
        f.write('kSVM - Classification report - Training set \n')
        f.write(classification_report(y_train, y_train_pred) + '\n')
        y_test_pred = ksvm.predict(X_test)
        f.write('kSVM - Classification report - Test set \n')
        f.write(classification_report(y_test, y_test_pred) + '\n')

        acc_test = accuracy_score(y_test, y_test_pred)
        f1_test = f1_score(y_test, y_test_pred)
        acc.append(round(acc_test,3))
        f1.append(round(f1_test,3))
        f.write(' & '+str(round(acc_test,3))+' & '+str(round(f1_test,3))+'\\\\')

print('acc = '+str(acc))
print('f1 = '+str(f1))
'''

data_list = ['liver','blood_transfusion','flowmeters','heart','diabetes','breast','divorce','australian','Mesothelioma','Gallstone','sonar','breast_wisconsin','germannumer','Fertility'] #'HillValley_training'
acc = [0.621,0.7,0.444,0.556,0.714,0.92,1.0,0.732,0.708,0.578,0.524,0.877,0.7,0.9]
f1 = [0.154,0.118,0.167,0.0,0.796,0.871,1.0,0.755,0.829,0.597,0.63,0.91,0.0,0.0]
acc_ksvm = [0.621, 0.767, 0.778, 0.815, 0.799, 0.993, 1.0, 0.891, 1.0, 0.828, 0.905, 0.974, 0.7, 0.9] #HillValley_training = 0.689
f1_ksvm = [0.353, 0.103, 0.6, 0.783, 0.858, 0.99, 1.0, 0.882, 1.0, 0.831, 0.9, 0.979, 0.0, 0.0] #HillValley_training = 0.578

differences_acc = np.abs(np.array(acc_ksvm)-np.array(acc))
max_diff_acc = np.max(differences_acc)
min_diff_acc = np.min(differences_acc)
print('max acc diff = '+str(round(max_diff_acc,3)))
print('min acc diff = '+str(round(min_diff_acc,3)))
print('Accuracy diff = '+str(differences_acc))

differences_f1 = np.abs(np.array(f1_ksvm)-np.array(f1))
max_diff_f1 = np.max(differences_f1)
min_diff_f1 = np.min(differences_f1)
print('max f1 diff = '+str(round(max_diff_f1,3)))
print('min f1 diff = '+str(round(min_diff_f1,3)))
print('F1-score diff = '+str(differences_f1))

quit()

plt.figure()
sc_acc = plt.plot(data_list, acc, marker='o', color='red', label='Spherical Classifier')
ksvm_acc = plt.plot(data_list, acc_ksvm, marker='o', color='blue', label='Kernel SVM')
plt.xticks(range(len(data_list)),data_list,rotation=90)
plt.ylim(-0.1, 1.1)
plt.xlabel('datasets')
plt.ylabel('accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('svm_experiments/accuracy_sc_vs_ksvm.pdf')

plt.figure()
sc_f1 = plt.plot(data_list, f1, marker='o', color='red', label='Spherical Classifier')
ksvm_f1 = plt.plot(data_list, f1_ksvm, marker='o', color='blue', label='Kernel SVM')
plt.xticks(range(len(data_list)),data_list,rotation=90)
plt.ylim(-0.1, 1.1)
plt.xlabel('datasets')
plt.ylabel('F1-score')
plt.legend()
plt.tight_layout()
plt.savefig('svm_experiments/f1_sc_vs_ksvm.pdf')