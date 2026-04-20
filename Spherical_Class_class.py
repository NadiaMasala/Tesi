#PROGETTO MACHINE LEARNING - Nadia Masala
#CLASSIFICATORE SFERICO

from sklearn.base import BaseEstimator , ClassifierMixin
from Helper_SC import *


class Spherical_Classifier( BaseEstimator , ClassifierMixin ) :

    def __init__(self, epsilon = 0.2, C1 = 1.0, C2 = 1.0, center = 'free'):
        self.epsilon = epsilon
        self.C1 = C1
        self.C2 = C2
        self.center = center

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        classes = np.unique(y)
        self.classes_ = classes

        if self.center == 'fixed':
            r, xi_neg, xi_pos = spherical_class_fit_semidef(self.X_, self.y_, self.C1, self.C2)
            self.r_ = r
            self.c_ = np.zeros(self.X_.shape[1])
            self.xi_neg_ = xi_neg
            self.xi_pos_ = xi_pos
        elif self.center == 'free':
            r, c, xi_neg, xi_pos = spherical_class_fit_semidef2(self.X_, self.y_, self.C1, self.C2)
            self.r_ = r
            self.c_ = c
            self.xi_neg_ = xi_neg
            self.xi_pos_ = xi_pos

        return self


    def predict(self, X_test):
        if self.center == 'fixed':
            y_pred = spherical_class_pred(X_test, self.r_)
        elif self.center == 'free':
            y_pred = spherical_class_pred2(X_test, self.r_, self.c_)

        return y_pred
