#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: DPP Utilities
####
####  Last updated: 5/4/16
####
####  Notes and disclaimers:
####    - Use only numpy.ndarray, not numpy.matrix to avoid any confusion
####    - If something is a column vector - MAKE IT A COLUMN VECTOR. Makes 
####          manipulation annoying, but it keeps matrix algebra logical. 
####
#########################################################################


#########################################################################
###
### IMPORTS
###

import sys
import numpy as np
import numpy.matlib as matlib
import scipy.linalg as linalg

#########################################################################


#########################################################################
###
### COLUMN_GAMMA_ZERO
###
### Last Updated: 3/5/16
###

def columnGammaZero(X,gamma):
    """
        Params: Gamma is the inclusion vector indexing columns of X
        Output: matrix with columns X_i if gamma_i = 1, column of 
                zeros if gamma_i = 0.
    """
    # Verify types of inputs and that gamma has length features, (columns of X)
    assert(type(X) == np.ndarray)
    assert(type(gamma) == np.ndarray)
    assert(gamma.shape == (X.shape[1],1))
    n, p = X.shape
    Xgam = np.array([X[:,i] if gamma[i,0] else np.zeros((n,)) for i in range(p)]).T
    return Xgam
    

#########################################################################


#########################################################################
###
### GAMMA_ZERO_2D
###
### Last Updated: 3/5/16
###

def gammaZero2D(X,gamma):
    """
        Params: Gamma is the inclusion vector indexing columns and
                rows of X
        Output: returns X with rows and columns of zeros for gamma_i = 0
    """
    # Verify types of inputs, that X is square, and gamma is the same length
    assert(type(X) == np.ndarray)
    assert(len(X.shape) == 2)
    assert(X.shape[0] == X.shape[1])
    assert(type(gamma) == np.ndarray)
    assert(gamma.shape == (X.shape[0],1))
    n = X.shape[0]
    Xcol = np.array([X[:,i] if gamma[i,0] else np.zeros((n,)) for i in range(n)])
    Xgam = np.array([Xcol[:,i] if gamma[i,0] else np.zeros((n,)) for i in range(n)])
    return Xgam
    
    

#########################################################################


#########################################################################
###
### GAMMA_RM_2D
###
### Last Updated: 3/5/16
###

def gammaRM2D(X,gamma):
    """
        Params: Gamma is the inclusion vector indexing columns and
                rows of X
        Output: returns X, removing rows and columns for gamma_i = 0
    """
    # Verify types of inputs, that X is square, and gamma is the same length
    assert(type(X) == np.ndarray)
    assert(len(X.shape) == 2)
    assert(X.shape[0] == X.shape[1])
    assert(type(gamma) == np.ndarray)
    assert(gamma.shape == (X.shape[0],1))
    n = X.shape[0]
    Xcol = np.array([X[:,i] for i in range(n) if gamma[i,0]])
    Xgam = np.array([Xcol[:,i] for i in range(n) if gamma[i,0]])
    return Xgam
    
    

#########################################################################


#########################################################################
###
### REMOVE_RC
###
### Last Updated: 4/6/16
###

def addback_RC(X,gamma):
    """
        Params: X is a square matrix, and gamma indexes the columns that 
                X represents of a larger matrix
        Output: Returns a larger matrix, with X on its rows and columns
                according to gamma.
    """

    assert(type(X) == np.ndarray)
    assert(len(X.shape) == 2)
    assert(X.shape[0] == X.shape[1])
    assert(X.shape[0] == sum(gamma))

    n = gamma.shape[0]
    m = X.shape[0]

    j = 0
    Xr = np.zeros((m,n))
    for i in range(n):
        if gamma[i]:
            Xr[:,i] = X[:,j]
            j += 1
        else:
            Xr[:,i] = np.zeros((m,))

    Xrc = np.zeros((n,n))
    j = 0
    for i in range(n):
        if gamma[i]:
            Xrc[i,:] = Xr[j,:]
            j += 1
        else:
            Xrc[i,:] = np.zeros((n,))
    return Xrc

#########################################################################



#########################################################################
###
### GETK_DIAG
###
### Last Updated: 3/18/16
###

def getKDiag(L):
    """
        Params: None - Assume exp(theta/2)X^TXexp(theta/2) form of L
        Output: Column Vector of diag(K)
    """
    assert(type(L) == np.ndarray)
    assert(L.shape[0] == L.shape[1])
    eigVals, eigVecs = linalg.eigh(L)
    scale = matlib.repmat(eigVals / (eigVals + 1.0), len(eigVals),1)

    # Pointwise product here leaves A_ij = v_j(i)^2 * lam_j / (lam_j + 1)
    sqrAndScale = eigVecs * eigVecs * scale
    # Sum up rows to get diagonal 
    Kdiag = np.array([np.sum(sqrAndScale,axis=1)]).T 

    return Kdiag

#########################################################################


#########################################################################
###
### GETK
###
### Last Updated: 3/18/16
###

def getK(L):
    """
        Params: None - Assume exp(theta/2)X^TXexp(theta/2) form of L
        Output: K
    """
    assert(type(L) == np.ndarray)
    assert(L.shape[0] == L.shape[1])
    n = L.shape[0]
    K = np.eye(n) - linalg.inv(L + np.eye(n))
    return K

#########################################################################



#########################################################################
###
### MAKE_L
###
### Last updated: 4/25/16
###
###

def makeL(S,theta):
    assert(type(S) == np.ndarray)
    assert(type(theta) == np.ndarray)
    assert(S.shape == (theta.shape[0],theta.shape[0]))

    expTheta = np.exp(0.5*theta)
    coeffs = expTheta.dot(expTheta.T)
    L = coeffs * S
    return L

#########################################################################



#########################################################################
###
### LDPP_MAKE_L
###
### Last updated: 5/4/16
###
###

def LDPP_makeL(S,theta,w):
    assert(type(S) == np.ndarray)
    assert(theta >= 0.0)

    L = np.exp(w) * (np.exp(-1 * theta) * S + (1.0 - np.exp(-1 *theta)) * np.eye(S.shape[0]))
    return L

#########################################################################



#########################################################################
###
### GREEDY_MAP_ESTIMATE
###
### Last Updated: 3/19/16
###

def greedyMapEstimate(p,L):
    """
        Params: p is the dimensionality of the inclusion vector, L is a
                likelihood function on gamma.
        Output: MAP estimate of gamma
    """
    assert(type(p) == int)
    
    gamma = np.zeros((p,1))
    U = range(0,p)#map(lambda e: np.array([e]).T, np.eye(p))

    L_old = L(gamma)
    # print gamma
    max = L(gamma)
    # print "max: %s" % repr(max)

    while len(U) > 0:
        idx = -1
        test = gamma
        nextChecks = zip(range(len(U)),U)
        np.random.shuffle(nextChecks)
        for n,e in nextChecks:
            test[e,0] = 1.0
            L_test = L(test)
            # print "GAMMA SENT: %s " % repr(test)
            # print "L_TEST: %s  for gamma: %s" % (repr(L_test), repr(test))
            if L_test > max:
                max = L_test
                idx = n
            test[e,0] = 0.0
        if idx >= 0:
            gamma[U[idx],0] = 1.0
            U.pop(idx)

        else:
            return gamma

    return gamma



#########################################################################

if __name__ == '__main__':
    X = np.array([[1,2,3],[4,5,6],[7,8,9]])
    gam = np.array([[1,0,1,0,0,1]]).T
    print addback_RC(X,gam)