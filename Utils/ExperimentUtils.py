#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Experiment Utilitiex
####
####  Last updated: 4/24/16
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

import os
import sys
mainpath = "/Users/Ted/__Engelhardt/Code"
sys.path.append(os.path.abspath(mainpath))
import time
import datetime

import numpy as np
import skmonaco as skmonaco
import scipy.optimize as optimize
import scipy.linalg as linalg
import random as random
import itertools as itertools

from Experiments.VI import VI

#########################################################################


#########################################################################
###
### GRID_SEARCH_1D
###
### Last Updated: 4/7/16
###

def gridSearch1D(grid, Learn, Eval, MAX = False, verbose = False):

    if MAX:
        best = sys.float_info.min 
    else:
        best = sys.float_info.max 

    bestVal = grid[0]

    for var in grid:
        L = Learn(var)
        current = Eval(L)
        while type(current) == np.ndarray:
            current = current[0]
        if MAX:
            if current > best:
                best = current
                bestVal = var
        else:
            if current < best:
                best = current
                bestVal = var

        if verbose:
            print "Current Loss: %f   for Lambda: %f" % (current, var)

    return bestVal


#########################################################################


#########################################################################
###
### DPP_SAMPLER
###
### Last Updated: 4/14/16
###

def DPPSampler(eigVals, eigVecs, indices = False):

    # Build set J with elements i from [1, M] w.p. lam_i / (lam_i + 1)
    J = eigSample(eigVals)
    V = np.array([eigVecs[:, i] for i in J]).T
    Y = []

    # Select elements of Y, with probability as averaged sum of squares of
    # ith component of eivenvectors in V. 
    while len(V) > 0:
        N = V.shape[0]
        n = V.shape[1]
        PDF = np.array([V[i, :].T.dot(V[i,:]) / float(n) for i in range(0,N)])
        pDist = np.cumsum(PDF)

        r = random.random()
        i = len([x for x in pDist if x < r])
        Y.append(i)
        V = orthogonalize(V, i)


    # Return Y as a numpy array column vector
    if indices:
        return np.array([Y]).T
    else:
        Y_Lspace = np.zeros((len(eigVals),1))
        for i in range(0,len(Y)):
            Y_Lspace[Y[i]] = 1.0
        return Y_Lspace

#########################################################################


#########################################################################
###
### ORTHOGONALIZE
###
### Last updated: 11/03/15
###
### Note: Given a set of R basis vectors and i \in [0, N-1], compute a 
###       new set of R-1 vectors that are orthogonal to e_i, but span the
###       rest of the original space spanned by the R original vectors. 
###

def orthogonalize(basis, i):
    """
     Params: basis is an orthonormal basis for some k-dimensional subspace
             of R^N. i indicates the dimension within N to which the 
             produced basis should be orthogonal. 
     Result: a new set of basis vectors, orthogonal to e_i, but otherwise
             spanning the original space. 
    """

    # Check type of eigVals
    assert type(basis) == np.ndarray

    N = basis.shape[0]
    d = basis.shape[1]
    dimNull = N - d + 1 # dimension of null space of newBasis

    if d == 1:
        return np.array([])

    eps = 1.0e-8 # Don't want to be dividing by some very small number
    # Get first column of basis with nonzero entry in ith row
    j = 0 # column number
    while j < d:
        if abs(basis[i][j]) < eps:
            j += 1
        else:
            break
    
    # kth column minus the (j) scaling column times ratio of ith element in kth to jth
    newOrthogonal = np.array([basis[:,k] - basis[:,j] * (float(basis[i,k]) / basis [i,j]) for k in range(0,basis.shape[1]) if k != j]).T
    newOrthonormal, R = linalg.qr(newOrthogonal)


    # Since we reduced the span of the basis, QR should have a larger nullspace (dim = n), given
    # by n rows of zeros in R. In Python's implementation of QR decomposition, these should
    # appear in the final n rows. Verify that this is the case. We verify this when we remove the
    # corresponding n rightmost columns of Q
    for nullCheck in range(0, dimNull):
        assert sum(R[N - nullCheck - 1, :]) == 0
    V = np.array(newOrthonormal[:,0:N - dimNull])

    return V


#########################################################################


#########################################################################
###
### EIG_SAMPLE
###
### Last updated: 11/02/15
###
### Note: Given a set of values k_i, we return a set of indices, each 
###       index i is included in the set w.p. k_i / (k_i + 1)
###

def eigSample(eigVals):
    """
     Params: eigVals is a set of eigenValues.   
     Result: a set of indices, each index i is included in the set 
             w.p. lam_i / (lam_i + 1)
    """

    # Check type of eigVals
    assert type(eigVals) == np.ndarray

    M = len(eigVals)

    # convert eigVals to probabilities
    probs = np.array([lam / (1.0 + lam) for lam in eigVals])

    # Create inclusion vector
    indices = np.array([i for i in range(0, M) if probs[i] > random.random()])

    return indices
    
# def testEigSample(N):
#     eigVals = np.array(random.rand(20,1))
#     freq = np.zeros(eigVals.shape)
#     for i in range(0,N):
#         indices = eigSample(eigVals)
#         for j in range(0, len(indices)):
#             freq[indices[j]] += 1.0/N
#     probs = np.array([lam / (1.0 + lam) for lam in eigVals])
#     print probs - freq
    

#########################################################################




#########################################################################
###
### C_RR_OPT
###
### Last Updated: 4/24/16
###
### Note: Returns the optimal value for the regularization parameter in
###       ridge regression. Note - we're not splitting into a train and
###       validation set yet.
###

def cRROpt(X, y, n=20,lam_min=-2,lam_max=3):
    val_size = int(0.1 * X.shape[0])
    X_val = X[0:val_size,:]
    y_val = y[0:val_size,:]
    X_train = X[val_size:,:]
    y_train = y[val_size:,:]

    def Eval(learned):
        learned_yhat = X_val.dot(learned)
        learned_mse = sum((y_val - learned_yhat) ** 2)
        return learned_mse

    def Learn(lam):
        inverse = linalg.inv(X_train.T.dot(X_train) + lam * np.eye(X.shape[1]))
        learned_beta = inverse.dot(X_train.T).dot(y_train)
        return learned_beta

    lams = np.logspace(lam_min,lam_max,n)

    opt_c = gridSearch1D(lams,Learn,Eval)

    return opt_c


#########################################################################



#########################################################################
###
### INIT_THETA
###
### Last updated: 11/02/15
###
### Note: Calculate starting theta: corresponds to line 3 in Variational 
###       Learning for Diverse VS in my notes. Given second derivative,
###       sciPy.optimize.newton uses Halley's parabolic method to solve
###       for e^theta0.
###

def initTheta(eigVals, kappa, expThetaGuess = 1.0, tolerance = 1.0e-12):
    """
     Params: eigVals is a numpy.ndarray of all eigenvalues of Phi*Phi^T
             k is the desired cardinality of the DPP
     Result: theta0 is the starting value of parameter theta. 
    """
    assert type(eigVals) == np.ndarray # Check type of eigVals
    assert (kappa >= 0)                # Verify cardinality is positive

    # Define the function and its first two derivatives for solving
    f            = lambda x: sum([(x*lam) / (1.0 + x*lam) for lam in eigVals]) - kappa
    fPrime       = lambda x: sum([lam / (1.0 + x*lam) ** 2 for lam in eigVals])
    fDoublePrime = lambda x: sum([-2.0 * lam ** 2 / (1.0 + x * lam) ** 3 for lam in eigVals])

    expTheta0 = optimize.newton(func = f, x0 = expThetaGuess, fprime = fPrime, fprime2 = fDoublePrime, tol = tolerance)
    theta0 = np.log(expTheta0)

    return theta0


#########################################################################

