# Spherical Classifier with selection of class in the sphere

from sklearn.base import BaseEstimator , ClassifierMixin
from New_Helper_SC import *


class New_Spherical_Classifier( BaseEstimator , ClassifierMixin ) :

    def __init__(self, epsilon = 0.2, minpts = 3, C1 = 1.0, C2 = 1.0, center = 'free'):
        self.epsilon = epsilon
        self.minpts = minpts
        self.C1 = C1
        self.C2 = C2
        self.center = center

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y

        labels = np.unique(y)
        self.labels_ = labels

        if self.center == 'fixed':
            r, xi_in, xi_out, in_class, out_class, in_label, out_label = new_spherical_class_fit_semidef(self.X_, self.y_, self.epsilon, self.C1, self.C2)
            #r, xi_in, xi_out, in_class, out_class, in_label, out_label = new2_spherical_class_fit_semidef(self.X_,self.y_,self.epsilon,self.minpts,self.C1,self.C2)
            self.r_ = r
            self.c_ = np.zeros(self.X_.shape[1])
            self.xi_in_ = xi_in
            self.xi_out_ = xi_out
            self.in_class_ = in_class
            self.out_class_ = out_class
            self.in_label_ = in_label
            self.out_label_ = out_label
        elif self.center == 'free':
            r, c, xi_in, xi_out, in_class, out_class, in_label, out_label = new_spherical_class_fit_semidef2(self.X_, self.y_, self.epsilon, self.C1, self.C2)
            #r, c, xi_in, xi_out, in_class, out_class, in_label, out_label = new2_spherical_class_fit_semidef2(self.X_,self.y_,self.epsilon,self.minpts,self.C1,self.C2)
            self.r_ = r
            self.c_ = c
            self.xi_in_ = xi_in
            self.xi_out_ = xi_out
            self.in_class_ = in_class
            self.out_class_ = out_class
            self.in_label_ = in_label
            self.out_label_ = out_label

        return self


    def predict(self, X_test):
        if self.center == 'fixed':
            y_pred = new_spherical_class_pred(X_test, self.r_, self.in_label_, self.out_label_)
        elif self.center == 'free':
            y_pred = new_spherical_class_pred2(X_test, self.r_, self.c_, self.in_label_, self.out_label_)

        return y_pred

