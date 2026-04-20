#PROGETTO MACHINE LEARNING - Nadia Masala
#Main per CLASSIFICAZIONE SFERICO - datasets reali

import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, davies_bouldin_score, silhouette_score
from Spherical_Class_class import Spherical_Classifier
from sklearn.svm import SVC

dataset = 4

if dataset == 1:
    train_set = pd.read_csv("Breast_train.csv",sep=';')
    test_set = pd.read_csv("Breast_test.csv",sep=';')

    #print("Le features sono:", train_set.columns.tolist())
    #for col in train_set.columns:
    #    print(f"Valori possibili per '{col}':", train_set[col].unique())
    #print("-----------------------------------------------------------")
    #for col in test_set.columns:
    #    print(f"Valori possibili per '{col}':", test_set[col].unique())
    #print("-----------------------------------------------------------")

    # Estrazione di X_train, X_test, y_train, y_test dai dataframe di pandas
    X_train = train_set.iloc[:,:-1]
    X_test = test_set.iloc[:,:-1]
    y_train = train_set.iloc[:,-1]
    y_test = test_set.iloc[:,-1]

    # Conversione dei dati categoriali in numeri interi
    ordinal_enc = OrdinalEncoder()
    X_train = ordinal_enc.fit_transform(X_train)
    X_test = ordinal_enc.transform(X_test)

    # Conversione etichette categoriali in formato binario
    y_train = label_binarize(y_train, classes=['no','yes'], neg_label=-1, pos_label=1)
    y_test = label_binarize(y_test, classes=['no','yes'], neg_label=-1, pos_label=1)

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test)).ravel()

elif dataset == 2:
    sample_set = pd.read_csv("Australian.csv")
    label_set = pd.read_csv("Australian_label.csv")
    X = sample_set.values
    y = label_set.iloc[:,0]
    y = label_binarize(y, classes=[-1,1], neg_label=-1, pos_label=1).ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

elif dataset == 3:
    sample_set = pd.read_csv("Diabetes.csv")
    label_set = pd.read_csv("Diabetes_label.csv")
    X = sample_set.values
    y = label_set.iloc[:,0]
    y = label_binarize(y, classes=[-1,1], neg_label=-1, pos_label=1).ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

elif dataset == 4:
    sample_set = pd.read_csv("Liver.csv")
    label_set = pd.read_csv("Liver_label.csv")
    X = sample_set.values
    y = label_set.iloc[:,0]
    y = label_binarize(y, classes=[0,1], neg_label=-1, pos_label=1).ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


m = X.shape[0]
n = X.shape[1]

print("Il dataset ha numero totale di elementi:",m)
print("Il dataset ha numero di features:",n)

# Selezione degli iperparametri (Grid Search)
C1_par = [2, 5, 10, 15, 20]
C2_par = [2, 5, 10, 15, 20]
center_par = ['fixed','free']
selected_parameters = {'C1':C1_par, 'C2':C2_par, 'center':center_par}
sc_grid = GridSearchCV(Spherical_Classifier(), selected_parameters, cv=5, verbose = 10, n_jobs = 5)
sc_grid.fit(X_train, y_train)
best_params = sc_grid.best_params_
print(best_params)

# Classificazione Sferica
sc = Spherical_Classifier(C1 = best_params['C1'], C2 = best_params['C2'], center = best_params['center'])
sc.fit(X_train, y_train)
y_pred = sc.predict(X_train)
print(classification_report(y_train, y_pred))
y_pred_sc = sc.predict(X_test)
print(classification_report(y_test, y_pred_sc))
accuracy_sc = accuracy_score(y_test,y_pred_sc)
f1_weighted_sc = f1_score(y_test, y_pred_sc, average='weighted')

# SVM
svm = SVC(kernel='linear')  # default C=1.0
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test,y_pred_svm)
f1_weighted_svm = f1_score(y_test, y_pred_svm, average='weighted')

# SVM con kernel polinomiale
svmk = SVC(kernel='poly')  # default C=1.0, degree=3, coef0=0.0
svmk.fit(X_train, y_train)
y_pred_svmk = svmk.predict(X_test)
accuracy_svmk = accuracy_score(y_test,y_pred_svmk)
f1_weighted_svmk = f1_score(y_test, y_pred_svmk, average='weighted')

methods = ['Classificazione Sferica', 'SVM', 'Kernel SVM']
accuracy = [accuracy_sc, accuracy_svm, accuracy_svmk]
f1_score = [f1_weighted_sc, f1_weighted_svm, f1_weighted_svmk]

# Confronto tra i classificatori
plt.bar(methods, accuracy, color=["red", "blue", "green"])
plt.ylabel("accuracy_score")
plt.title("Istogramma accuratezza dei classificatori")
plt.ylim(0, 1)
plt.savefig('C:/Users/nadia_2/Desktop/UNIVERSITA/MAGISTRALE/Machine Learning/Progetto/Liver_acc.png', bbox_inches='tight')
plt.show()

plt.bar(methods, f1_score, color=["red", "blue", "green"])
plt.ylabel("f1_weighted_score")
plt.title("Istogramma f1-score pesato dei classificatori")
plt.ylim(0, 1)
plt.savefig('C:/Users/nadia_2/Desktop/UNIVERSITA/MAGISTRALE/Machine Learning/Progetto/Liver_f1.png', bbox_inches='tight')
plt.show()




