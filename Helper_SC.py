#PROGETTO MACHINE LEARNING - Nadia Masala
#Helper functions per CLASSIFICATORE SFERICO

import numpy as np
import cvxpy as cp

# Fit con centro nell'origine
def spherical_class_fit_semidef(X, y, C1, C2):
    m = X.shape[0]
    n = X.shape[1]

    y_temp = np.copy(y)

    # Suddivisione dei punti in input in base alle etichette vere
    X_neg = []
    X_pos = []
    for i in range(m):
        if y[i] == -1:
            X_neg.append(X[i])
        elif y[i] == +1:
            X_pos.append(X[i])
    X_neg = np.array(X_neg)
    X_pos = np.array(X_pos)

    # Definizione delle variabili
    Q = cp.Variable((n,n),symmetric=True)
    xi_neg = cp.Variable(X_neg.shape[0])
    xi_pos = cp.Variable(X_pos.shape[0])

    # Definizione dei vincoli
    constr = []
    for i in range(X_neg.shape[0]):
        constr += [X_neg[i] @ Q @ X_neg[i].T <= 1 + xi_neg[i] , xi_neg[i] >= 0]
    for j in range(X_pos.shape[0]):
        constr += [X_pos[j] @ Q @ X_pos[j].T >= 1 - xi_pos[j], xi_pos[j] >= 0]
    for i in range(n):
        for j in range(n):
            constr += [Q[i,i] == Q[j,j]]  # Q di entrate uguali lungo la diagonale principale
            if i != j:
                constr += [Q[i,j] == 0]  # Q diagonale
    constr += [Q >> 0]

    # Definizione della funzione obiettivo e del tipo di problema
    obj = Q[0,0] - C1 * cp.sum(xi_neg) - C2 * cp.sum(xi_pos)
    objective = cp.Maximize(obj)

    # Risoluzione del problema
    prob = cp.Problem(objective,constr)
    res = prob.solve(solver = cp.MOSEK, verbose = 0)

    # Salvataggio delle soluzioni
    Q_star = Q.value
    r_star = np.sqrt(1/Q_star[0,0])
    xi_neg_star = xi_neg.value
    xi_pos_star = xi_pos.value

    return r_star, xi_neg_star, xi_pos_star

# Predict con centro nell'origine
def spherical_class_pred(X_test, r):
    m = X_test.shape[0]
    y_pred = []

    for i in range(m):
        if (np.linalg.norm(X_test[i]))**2 - r**2 <= 0:
            y_pred.append(-1)
        else:
            y_pred.append(+1)

    return np.array(y_pred)

# Fit con centro libero
def spherical_class_fit_semidef2(X, y, C1, C2):
    m = X.shape[0]
    n = X.shape[1]

    y_temp = np.copy(y)

    # Ridefinizione in R^(n+1) dei punti in input
    Xx = np.hstack((np.ones((m, 1)), X))  # (m,n+1)

    # Suddivisione dei punti in base alle etichette vere
    Xx_neg = []
    Xx_pos = []
    for i in range(m):
        if y[i] == -1:
            Xx_neg.append(Xx[i])
        elif y[i] == +1:
            Xx_pos.append(Xx[i])
    Xx_neg = np.array(Xx_neg)
    Xx_pos = np.array(Xx_pos)

    # Definizione delle variabili
    Q_tilde = cp.Variable((n + 1, n + 1), symmetric=True)
    F = Q_tilde[1:, 1:]  # sottomatrice di Q (n,n)
    xi_neg = cp.Variable(Xx_neg.shape[0])
    xi_pos = cp.Variable(Xx_pos.shape[0])

    # Definizione dei vincoli
    constr = []
    for i in range(Xx_neg.shape[0]):
        constr += [Xx_neg[i] @ Q_tilde @ Xx_neg[i].T <= 1 + xi_neg[i], xi_neg[i] >= 0]
    for j in range(Xx_pos.shape[0]):
        constr += [Xx_pos[j] @ Q_tilde @ Xx_pos[j].T >= 1 - xi_pos[j], xi_pos[j] >= 0]
    for i in range(n):
        for j in range(n):
            constr += [F[i, i] == F[j, j]]  # F di entrate uguali lungo la diagonale principale
            if i != j:
                constr += [F[i, j] == 0]  # F diagonale
    constr += [Q_tilde >> 0]  # Q_tilde semidefinita positiva

    # Definizione della funzione obiettivo e del tipo di problema
    obj = Q_tilde[0, 0] - C1 * cp.sum(xi_neg) - C2 * cp.sum(xi_pos)  # - per minimizzare le slack
    objective = cp.Maximize(obj)

    # Risoluzione del problema
    prob = cp.Problem(objective, constr)
    res = prob.solve(solver=cp.MOSEK, verbose=0)

    # Salvataggio delle soluzioni
    Q_tilde_star = Q_tilde.value
    F_star = Q_tilde_star[1:, 1:]
    t_star = Q_tilde_star[0, 1:]
    s_star = Q_tilde_star[0, 0]
    c_star = - np.linalg.inv(F_star) @ t_star  # centro della sfera ottimale
    delta_star = s_star - c_star @ F_star @ c_star.T
    Q_star = F_star / (1 - delta_star)
    r_star = np.sqrt(1 / Q_star[0, 0])
    xi_neg_star = xi_neg.value
    xi_pos_star = xi_pos.value

    return r_star, c_star, xi_neg_star, xi_pos_star

# Predict con centro libero
def spherical_class_pred2(X_test, r, c):
    m = X_test.shape[0]
    y_pred = []

    for i in range(m):
        if (np.linalg.norm(X_test[i] - c)) ** 2 - r ** 2 <= 0:
            y_pred.append(-1)
        else:
            y_pred.append(+1)

    return np.array(y_pred)



