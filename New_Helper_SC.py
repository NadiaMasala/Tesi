# Helper functions for Spherical Classification with selection of class of label '1'

import numpy as np
import cvxpy as cp
from sklearn.preprocessing import label_binarize

# Fit with fixed center in the origin
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

# Predict with fixed center in the origin
def new_spherical_class_pred(X_test, r, in_label, out_label):
    m = X_test.shape[0]
    y_pred = []

    for i in range(m):
        if (np.linalg.norm(X_test[i]))**2 - r**2 <= 0:
            y_pred.append(in_label)
        else:
            y_pred.append(out_label)

    return np.array(y_pred)

# Fit with free center
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

    # Definition fo constraints
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
    obj = Q_tilde[0, 0] - C1 * cp.sum(xi_in) - C2 * cp.sum(xi_out)
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

# Predict with free center
def new_spherical_class_pred2(X_test, r, c, in_label, out_label):
    m = X_test.shape[0]
    y_pred = []

    for i in range(m):
        if (np.linalg.norm(X_test[i] - c)) ** 2 - r ** 2 <= 0:
            y_pred.append(in_label)
        else:
            y_pred.append(out_label)

    return np.array(y_pred)

# Function for selection of class in the sphere
def class_in_selection(X, y, epsilon):
    m = X.shape[0]
    n = X.shape[1]

    labels = np.unique(y)

    A = []
    B = []
    for i in range(m):
        if y[i] == labels[0]:
            A.append(X[i])
        elif y[i] == labels[1]:
            B.append(X[i])
    A = np.array(A)
    B = np.array(B)

    # Definition of the barycenter of all points
    barycenter = np.zeros(n)
    for j in range(n):
        barycenter[j] = np.mean(X[:, j])

    distances = {}
    for i in range(m):
        distances[i] = np.linalg.norm(barycenter - X[i])

    A_in = []  # points of A in the sphere of radius = epsilon
    B_in = []  # points of B in the sphere of radius = epsilon
    for i in range(m):
        if distances[i] <= epsilon:
            if y[i] == labels[0]:
                A_in.append(X[i])
            elif y[i] == labels[1]:
                B_in.append(X[i])

    # Selection of the class that has to be in the separation sphere
    if len(A_in) >= len(B_in):
        in_class = A
        out_class = B
        in_label = labels[0]
        out_label = labels[1]
    else:
        in_class = B
        out_class = A
        in_label = labels[1]
        out_label = labels[0]

    return in_class, out_class, in_label, out_label

# Function 2 for selection of class in the sphere
def class_in_selection2(X, y, epsilon, minpts):
    m = X.shape[0]
    n = X.shape[1]

    labels = np.unique(y)

    A = []
    B = []
    for i in range(m):
        if y[i] == labels[0]:
            A.append(X[i])
        elif y[i] == labels[1]:
            B.append(X[i])
    A = np.array(A)
    B = np.array(B)

    # Defining classes centroids
    C_a = np.zeros(n)
    C_b = np.zeros(n)
    for j in range(n):
        C_a[j] = np.mean(A[:, j])
        C_b[j] = np.mean(B[:, j])

    distancesA = {}
    for i in range(A.shape[0]):
        distancesA[i] = np.linalg.norm(C_a - A[i])
    distancesB = {}
    for j in range(B.shape[0]):
        distancesB[j] = np.linalg.norm(C_b - B[j])

    A_in = []  # points of A in the sphere of radius = epsilon
    for i in range(A.shape[0]):
        if distancesA[i] <= epsilon:
                A_in.append(A[i])
    B_in = []  # points of B in the sphere of radius = epsilon
    for j in range(B.shape[0]):
        if distancesB[j] <= epsilon:
                B_in.append(B[j])

    # Selection of the class that has to be in the separation sphere
    if len(A_in) >= minpts or len(B_in) >= minpts:
        if len(A_in) >= len(B_in):
            in_class = A
            out_class = B
            in_label = labels[0]
            out_label = labels[1]
        else:
            in_class = B
            out_class = A
            in_label = labels[1]
            out_label = labels[0]


    return in_class, out_class, in_label, out_label

def new2_spherical_class_fit_semidef(X, y, epsilon, minpts, C1, C2):
    m = X.shape[0]
    n = X.shape[1]

    #Selection of class in
    X_in, X_out, in_label, out_label = class_in_selection2(X, y, epsilon, minpts)

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

def new2_spherical_class_fit_semidef2(X, y, epsilon, minpts, C1, C2):
    m = X.shape[0]
    n = X.shape[1]

    # New points in R^(n+1)
    Xx = np.hstack((np.ones((m, 1)), X))  # (m,n+1)

    # Selection of class in
    Xx_in, Xx_out, in_label, out_label = class_in_selection2(Xx, y, epsilon, minpts)

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
    obj = Q_tilde[0, 0] - C1 * cp.sum(xi_in) - C2 * cp.sum(xi_out)
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
