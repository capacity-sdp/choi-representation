import numpy as np

#returns the choi matrix representation for depolarizing channels
def choi_depol(q, d):
    Idd = (q/d) * np.eye(d * d)
    X = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            X[i,j] = 1
            Xtemp = np.kron(X, X)
            Idd += (1-q) * Xtemp
            X[i,j] = 0
    return Idd

#returns the choi matrix representation for Werner-Holevo channels
def choi_wern(p, d):
    Idd = (p/d) * np.eye(d * d)
    X = np.zeros((d, d))
    Xcopy = X.copy()
    for i in range(d):
        for j in range(d):
            X[i,j] = 1
            Xcopy[j, i] = 1
            Xtemp = np.kron(X, Xcopy)
            Idd += (1-p) * Xtemp
            X[i,j] = 0
            Xcopy[j, i] = 0
    return Idd
