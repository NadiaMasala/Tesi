import numpy as np
import cvxpy as cp
import sys
import mosek
from   mosek.fusion import *
from sklearn.preprocessing import label_binarize

def my_spherical_class_fit_semidef(X, y, epsilon, minpts, C1, C2):
    m = X.shape[0]
    n = X.shape[1]

    #Selection of class in
    X_in, X_out, in_label, out_label = my_class_in_selection(X, y, epsilon, minpts)

    # Definition of variables
    Q = cp.Variable((n,n),symmetric=True)
    xi_in = cp.Variable(X_in.shape[0])
    xi_out = cp.Variable(X_out.shape[0])

    # Definition of constraints
    constr = []
    for i in range(X_in.shape[0]):
        constr += [X_in[i] @ Q @ X_in[i].T <= 1 + xi_in[i] , xi_in[i] >= 0]
    for j in range(X_out.shape[0]):
        constr += [X_out[j] @ Q @ X_out[j].T >= 1 - xi_out[j], xi_out[j] >= 0]
    for i in range(n):
        for j in range(n):
            constr += [Q[i,i] == Q[j,j]]
            if i != j:
                constr += [Q[i,j] == 0]
    constr += [Q >> 0]

    # Objective function and optimization problem
    obj = Q[0,0] - C1 * cp.sum(xi_in) - C2 * cp.sum(xi_out)
    objective = cp.Maximize(obj)

    prob = cp.Problem(objective,constr)
    res = prob.solve(solver = cp.MOSEK, verbose = 0)

    # Solutions
    Q_star = Q.value
    r_star = np.sqrt(1/Q_star[0,0])
    xi_in_star = xi_in.value
    xi_out_star = xi_out.value

    return r_star, xi_in_star, xi_out_star, X_in, X_out, in_label, out_label

def my_spherical_class_fit_semidef2(X, y, epsilon, minpts, C1, C2):
    m = X.shape[0]
    n = X.shape[1]

    # New points in R^(n+1)
    Xx = np.hstack((np.ones((m, 1)), X))  # (m,n+1)

    # Selection of class in
    Xx_in, Xx_out, in_label, out_label = my_class_in_selection(Xx, y, epsilon, minpts)

    # Definition of variables
    Q_tilde = cp.Variable((n + 1, n + 1), symmetric=True)
    F = Q_tilde[1:, 1:]  # submatrix of Q (n,n)
    xi_in = cp.Variable(Xx_in.shape[0])
    xi_out = cp.Variable(Xx_out.shape[0])

    # Definition of constraints
    constr = []
    for i in range(Xx_in.shape[0]):
        constr += [Xx_in[i] @ Q_tilde @ Xx_in[i].T <= 1 + xi_in[i], xi_in[i] >= 0]
    for j in range(Xx_out.shape[0]):
        constr += [Xx_out[j] @ Q_tilde @ Xx_out[j].T >= 1 - xi_out[j], xi_out[j] >= 0]
    for i in range(n):
        for j in range(n):
            constr += [F[i, i] == F[j, j]]
            if i != j:
                constr += [F[i, j] == 0]  # F is a diagonal matrix
    constr += [Q_tilde >> 0]  # Q_tilde is semi-definite positive

    # Objective function and optimization problem
    obj = Q_tilde[1, 1] - C1 * cp.sum(xi_in) - C2 * cp.sum(xi_out)
    objective = cp.Maximize(obj)

    prob = cp.Problem(objective, constr)
    res = prob.solve(solver=cp.MOSEK, verbose=0)

    # Solutions
    Q_tilde_star = Q_tilde.value
    F_star = Q_tilde_star[1:, 1:]
    t_star = Q_tilde_star[0, 1:]
    s_star = Q_tilde_star[0, 0]
    c_star = - np.linalg.inv(F_star) @ t_star  # optimal center of the sphere
    delta_star = s_star - c_star @ F_star @ c_star.T
    Q_star = F_star / (1 - delta_star)
    r_star = np.sqrt(1 / Q_star[0, 0])
    xi_in_star = xi_in.value
    xi_out_star = xi_out.value

    X_in = np.delete(Xx_in,0,1)
    X_out = np.delete(Xx_out, 0, 1)

    return r_star, c_star, xi_in_star, xi_out_star, X_in, X_out, in_label, out_label

# da inizio file
def new_spherical_class_fit_semidef2(X, y, epsilon, C1, C2):
    m = X.shape[0]
    n = X.shape[1]

    # New points in R^(n+1)
    Xx = np.hstack((np.ones((m, 1)), X))  # (m,n+1)

    # Selection of class in
    Xx_in, Xx_out, in_label, out_label = class_in_selection(Xx, y, epsilon)

    # Definition of variables
    Q_tilde = cp.Variable((n + 1, n + 1), symmetric=True)
    F = Q_tilde[1:, 1:]  # submatrix of Q (n,n)
    xi_in = cp.Variable(Xx_in.shape[0])
    xi_out = cp.Variable(Xx_out.shape[0])

    # Definition of constraints
    constr = []
    for i in range(Xx_in.shape[0]):
        constr += [Xx_in[i] @ Q_tilde @ Xx_in[i].T <= 1 + xi_in[i], xi_in[i] >= 0]
    for j in range(Xx_out.shape[0]):
        constr += [Xx_out[j] @ Q_tilde @ Xx_out[j].T >= 1 - xi_out[j], xi_out[j] >= 0]
    for i in range(n):
        for j in range(n):
            constr += [F[i, i] == F[j, j]]
            if i != j:
                constr += [F[i, j] == 0]  # F is a diagonal matrix
    constr += [Q_tilde >> 0]  # Q_tilde is semi-definite positive

    # Objective function and optimization problem
    obj = Q_tilde[1, 1] - C1 * cp.sum(xi_in) - C2 * cp.sum(xi_out)
    objective = cp.Maximize(obj)

    prob = cp.Problem(objective, constr)
    res = prob.solve(solver=cp.MOSEK, verbose=1)

    # Solutions
    Q_tilde_star = Q_tilde.value
    F_star = Q_tilde_star[1:, 1:]
    t_star = Q_tilde_star[0, 1:]
    s_star = Q_tilde_star[0, 0]
    c_star = - np.linalg.inv(F_star) @ t_star  # optimal center of the sphere
    delta_star = s_star - c_star @ F_star @ c_star.T
    Q_star = F_star / (1 - delta_star)
    r_star = np.sqrt(1 / Q_star[0, 0])
    xi_in_star = xi_in.value
    xi_out_star = xi_out.value

    X_in = np.delete(Xx_in,0,1)
    X_out = np.delete(Xx_out, 0, 1)

    return r_star, c_star, xi_in_star, xi_out_star, X_in, X_out, in_label, out_label

def new_spherical_class_fit_semidef(X, y, epsilon, C1, C2):
    m = X.shape[0]
    n = X.shape[1]

    #Selection of class in
    X_in, X_out, in_label, out_label = class_in_selection(X, y, epsilon)

    # Definition of variables
    Q = cp.Variable((n,n),symmetric=True)
    xi_in = cp.Variable(X_in.shape[0])
    xi_out = cp.Variable(X_out.shape[0])

    # Definition of constraints
    constr = []
    for i in range(X_in.shape[0]):
        constr += [X_in[i] @ Q @ X_in[i].T <= 1 + xi_in[i] , xi_in[i] >= 0]
    for j in range(X_out.shape[0]):
        constr += [X_out[j] @ Q @ X_out[j].T >= 1 - xi_out[j], xi_out[j] >= 0]
    for i in range(n):
        for j in range(n):
            constr += [Q[i,i] == Q[j,j]]
            if i != j:
                constr += [Q[i,j] == 0]
    constr += [Q >> 0]

    # Objective function and optimization problem
    obj = Q[0,0] - C1 * cp.sum(xi_in) - C2 * cp.sum(xi_out)
    objective = cp.Maximize(obj)

    prob = cp.Problem(objective,constr)
    res = prob.solve(solver = cp.MOSEK, verbose = 1)

    # Solutions
    Q_star = Q.value
    r_star = np.sqrt(1/Q_star[0,0])
    xi_in_star = xi_in.value
    xi_out_star = xi_out.value

    return r_star, xi_in_star, xi_out_star, X_in, X_out, in_label, out_label