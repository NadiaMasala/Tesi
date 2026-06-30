# Helper functions for Spherical Classification with selection of class of label '1'

import numpy as np
import cvxpy as cp
import sys
import mosek
from   mosek.fusion import *
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

def spherical_class_fit_semidef_mosek(X, y, epsilon, minpts, C1, C2):
    m = X.shape[0]
    n = X.shape[1]

    #Selection of class in
    X_in, X_out, in_label, out_label = my_class_in_selection(X, y, epsilon, minpts)
    m_in = X_in.shape[0]
    m_out = X_out.shape[0]

    # Definition of variables
    with Model("model") as M:
        c1 = M.parameter()
        c1.setValue(C1)
        c2 = M.parameter()
        c2.setValue(C2)

        Q = M.variable(Domain.inPSDCone(n))
        xi_in = M.variable(m_in, Domain.greaterThan(0.0))
        xi_out = M.variable(m_out, Domain.greaterThan(0.0))

        # Objective function and optimization problem
        f_obj = Expr.sub(Q.index([0, 0]), Expr.add(Expr.mul(c1, Expr.sum(xi_in)), Expr.mul(c2, Expr.sum(xi_out))))
        M.objective(ObjectiveSense.Maximize, f_obj)

        # Constraints
        for i in range(m_in):
            xxt = X_in[i].reshape((n,1)) @ X_in[i].reshape((1,n))
            expr = Expr.dot(xxt,Q)
            M.constraint(Expr.sub(expr, Expr.add(1.0, xi_in.index(i))), Domain.lessThan(0.0))
        for i in range(m_out):
            xxt = X_out[i].T @ X_out[i]
            expr = Expr.dot(xxt, Q)
            M.constraint(Expr.sub(expr, Expr.sub(1.0, xi_out.index(i))), Domain.greaterThan(0.0))
        for i in range(n):
            for j in range(n):
                if i!=j:
                    M.constraint(Q.index([i,j]), Domain.equalsTo(0.0))
                else:
                    M.constraint(Expr.sub(Q.index([i,i]), Q.index([j,j])), Domain.equalsTo(0.0))

        M.setLogHandler(sys.stdout)
        M.solve()

        # Solutions
        Q_star = np.reshape(Q.level(),(n,n))
        r_star = np.sqrt(1/Q_star[0,0])
        xi_in_star = xi_in.level()
        xi_out_star = xi_out.level()

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

def spherical_class_fit_semidef2_mosek(X, y, epsilon, minpts, C1, C2):
    m = X.shape[0]
    n = X.shape[1]

    # New points in R^(n+1)
    Xx = np.hstack((np.ones((m, 1)), X))

    # Selection of class in
    Xx_in, Xx_out, in_label, out_label = my_class_in_selection(Xx, y, epsilon, minpts)
    m_in = Xx_in.shape[0]
    m_out = Xx_out.shape[0]


    with Model("model") as M:
        c1 = M.parameter()
        c1.setValue(C1)
        c2 = M.parameter()
        c2.setValue(C2)

        # Definition of variables
        Q_tilde = M.variable(Domain.inPSDCone(n+1))
        F = Q_tilde.slice([1,1],[n+1,n+1])
        xi_in = M.variable(m_in, Domain.greaterThan(0.0))
        xi_out = M.variable(m_out, Domain.greaterThan(0.0))

        # Objective function and optimization problem
        f_obj = Expr.sub(F.index([0,0]), Expr.add(Expr.mul(c1, Expr.sum(xi_in)), Expr.mul(c2, Expr.sum(xi_out))))
        M.objective(ObjectiveSense.Maximize, f_obj)

        # Constraints
        for i in range(m_in):
            xxt = Xx_in[i].reshape((n+1,1)) @ Xx_in[i].reshape((1,n+1))
            expr = Expr.dot(xxt,Q_tilde)
            M.constraint(Expr.sub(expr, Expr.add(1.0, xi_in.index(i))), Domain.lessThan(0.0))
        for i in range(m_out):
            xxt = Xx_out[i].reshape((n+1,1)) @ Xx_out[i].reshape((1,n+1))
            expr = Expr.dot(xxt, Q_tilde)
            M.constraint(Expr.sub(expr, Expr.sub(1.0, xi_out.index(i))), Domain.greaterThan(0.0))
        for i in range(n):
            for j in range(n):
                if i!=j:
                    M.constraint(F.index([i,j]), Domain.equalsTo(0.0))
                else:
                    M.constraint(Expr.sub(F.index([i,i]), F.index([j,j])), Domain.equalsTo(0.0))

        M.setLogHandler(sys.stdout)
        M.solve()

        # Solutions
        Q_tilde_star = np.reshape(Q_tilde.level(),(n+1,n+1))
        F_star = Q_tilde_star[1:,1:]
        t_star = Q_tilde_star[1:,0].reshape((n,1))
        s_star = Q_tilde_star[0,0]
        c_star = - np.linalg.inv(F_star) @ t_star
        delta_star = s_star - c_star.T @ F_star @ c_star
        Q_star = F_star / (1 - delta_star)
        r_star = np.sqrt(1 / Q_star[0,0])
        xi_in_star = xi_in.level()
        xi_out_star = xi_out.level()

        X_in = np.delete(Xx_in, 0, 1)
        X_out = np.delete(Xx_out, 0, 1)

    return r_star, c_star, xi_in_star, xi_out_star, X_in, X_out, in_label, out_label

def spherical_class_fit_semidef2_T_mosek(X, y, epsilon, minpts, C1, C2):
    m = X.shape[0]
    n = X.shape[1]

    # New points in R^(n+1)
    Xx = np.hstack((np.ones((m, 1)), X))

    # Selection of class in
    Xx_in, Xx_out, in_label, out_label = my_class_in_selection(Xx, y, epsilon, minpts)
    m_in = Xx_in.shape[0]
    m_out = Xx_out.shape[0]


    with Model("model") as M:
        c1 = M.parameter()
        c1.setValue(C1)
        c2 = M.parameter()
        c2.setValue(C2)

        # Definition of variables
        Q_tilde = M.variable(Domain.inPSDCone(n+1))
        F = Q_tilde.slice([1,1],[n+1,n+1])
        T = M.variable([n,n])
        Id = Matrix.eye(n)
        B = M.variable(Domain.inPSDCone(2*n))
        B11 = B.slice([0,0],[n,n])
        B12 = B.slice([0,n],[n,2*n])
        B21 = B.slice([n,0],[2*n,n])
        B22 = B.slice([n,n],[2*n,2*n])
        xi_in = M.variable(m_in, Domain.greaterThan(0.0))
        xi_out = M.variable(m_out, Domain.greaterThan(0.0))

        # Objective function and optimization problem
        f_obj = Expr.add(Expr.dot(T,Id), Expr.add(Expr.mul(c1, Expr.sum(xi_in)), Expr.mul(c2, Expr.sum(xi_out))))
        M.objective(ObjectiveSense.Minimize, f_obj)

        # Constraints
        for i in range(m_in):
            xxt = Xx_in[i].reshape((n+1,1)) @ Xx_in[i].reshape((1,n+1))
            expr = Expr.dot(xxt,Q_tilde)
            M.constraint(Expr.sub(expr, Expr.add(1.0, xi_in.index(i))), Domain.lessThan(0.0))
        for i in range(m_out):
            xxt = Xx_out[i].reshape((n+1,1)) @ Xx_out[i].reshape((1,n+1))
            expr = Expr.dot(xxt, Q_tilde)
            M.constraint(Expr.sub(expr, Expr.sub(1.0, xi_out.index(i))), Domain.greaterThan(0.0))
        for i in range(n):
            for j in range(n):
                if i!=j:
                    M.constraint(F.index([i,j]), Domain.equalsTo(0.0))
                else:
                    M.constraint(Expr.sub(F.index([i,i]), F.index([j,j])), Domain.equalsTo(0.0))
        M.constraint(Expr.sub(T, T.transpose()), Domain.equalsTo(0.0))
        M.constraint(Expr.sub(B11, F), Domain.equalsTo(0.0))
        M.constraint(Expr.sub(B12, Id), Domain.equalsTo(0.0))
        M.constraint(Expr.sub(B21, Id), Domain.equalsTo(0.0))
        M.constraint(Expr.sub(B22, T), Domain.equalsTo(0.0))

        M.setLogHandler(sys.stdout)
        M.solve()

        # Solutions
        Q_tilde_star = np.reshape(Q_tilde.level(),(n+1,n+1))
        T_star = np.reshape(T.level(),(n,n))
        F_star = Q_tilde_star[1:,1:]
        t_star = Q_tilde_star[1:,0].reshape((n,1))
        s_star = Q_tilde_star[0,0]
        c_star = - np.linalg.inv(F_star) @ t_star
        delta_star = s_star - c_star.T @ F_star @ c_star
        Q_star = F_star / (1 - delta_star)
        r_star = np.sqrt(1 / Q_star[0,0])
        xi_in_star = xi_in.level()
        xi_out_star = xi_out.level()

        X_in = np.delete(Xx_in, 0, 1)
        X_out = np.delete(Xx_out, 0, 1)

    return r_star, c_star.ravel(), xi_in_star, xi_out_star, X_in, X_out, in_label, out_label

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

# Function for selection of class inside the sphere
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

    A_in = []  # points of A inside the sphere of radius = epsilon
    B_in = []  # points of B inside the sphere of radius = epsilon
    for i in range(m):
        if distances[i] <= epsilon:
            if y[i] == labels[0]:
                A_in.append(X[i])
            elif y[i] == labels[1]:
                B_in.append(X[i])

    # Selection of the class that has to be inside the separation sphere
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

# My function for selection of class inside the sphere
def my_class_in_selection(X, y, epsilon, minpts):
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

    A_in = []  # points of A inside the sphere of radius = epsilon
    for i in range(A.shape[0]):
        if distancesA[i] <= epsilon:
                A_in.append(A[i])
    B_in = []  # points of B inside the sphere of radius = epsilon
    for j in range(B.shape[0]):
        if distancesB[j] <= epsilon:
                B_in.append(B[j])

    # Selection of the class that has to be inside the separation sphere
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
