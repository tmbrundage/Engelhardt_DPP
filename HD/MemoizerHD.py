#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Memoizer HD
####
####  Last updated: 4/21/16
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

import numpy as np 
import scipy.linalg as linalg
import operator as operator
import HD.DPPutilsHD as DPPutils

#########################################################################


class MemoizerHD(object):


    #########################################################################
    ###
    ### INITIALIZER
    ###
    ### Last Updated: 4/21/16
    ###

    def __init__(self, X, y, check = False):

        """
            Params:
             - X is the n x p matrix of normalized input data
             - y is the n x 1 column vector of normalized tartet values
        """
        if check:
            assert(type(X) == np.ndarray)
            assert(type(y) == np.ndarray)
            assert(X.shape[0] == y.shape[0])

        # Save data variables
        self.X = X
        self.y = y
        self.S = X.dot(X.T)
        self.check = check

        # Save constants
        self.n, self.p = X.shape

        # Create dictionaries
        self.differenceProjection = {} # 1/2 y.T (I - X_gam(X_gam^T X_gam + I)^-1 X_gam^T) y
        self.S_QLAM = {} # Eigendecomposition of S

    #########################################################################
   

    #########################################################################
    ###
    ### DET(S+Lam)
    ###
    ### Last Updated: 3/30/16
    ###

    def FdetSLam(self,gamma,c):
        """
            Params: >>>>> Gamma is the inclusion vector indexing S
            Output: det(S_gamma + cI)
        """

        if self.check:
            # Verify the type of gamma and that it is the right size and shape
            assert(type(gamma) == np.ndarray)
            assert(gamma.shape == (self.p,1))

        # If gamma is the empty set, determinant is just of I
        if (sum(gamma)[0] == 0.0):
            return c ** self.p

        key = str(gamma)
        
        # If we have already computed the eigenvalues, return them from the dictionary
        if self.S_QLAM.has_key(key):
            eigVals = self.S_QLAM.get(key)['eigVals']

        # Otherwise, find the eigendecomposition and memoize
        else:
            S_gamma = DPPutils.gammaZero2D(self.S,gamma)
            eigVals, eigVecs = linalg.eigh(S_gamma)
            self.S_QLAM[key] = {'eigVals': eigVals, 'eigVecs': eigVecs}

        det = reduce(operator.mul, map(lambda x: x + c, eigVals), 1.0)

        return det
        
    #########################################################################



    #########################################################################
    ###
    ### DET(S+Lam)
    ###
    ### Last Updated: 3/30/16
    ###

    def FinvSLam(self,gamma,c):
        """
            Params: >>>>> Gamma is the inclusion vector indexing S
            Output: inv(S_gamma + cI)
        """

        if self.check:
            # Verify the type of gamma and that it is the right size and shape
            assert(type(gamma) == np.ndarray)
            assert(gamma.shape == (self.p,1))

        # If gamma is the empty set, determinant is just of I
        if (sum(gamma)[0] == 0.0):
            return c ** self.p

        key = str(gamma)
        
        # If we have already computed the eigenvalues, return them from the dictionary
        if self.S_QLAM.has_key(key):
            eigVals = self.S_QLAM.get(key)['eigVals']
            eigVecs = self.S_QLAM.get(key)['eigVecs']

        # Otherwise, find the eigendecomposition and memoize
        else:
            S_gamma = DPPutils.gammaZero2D(self.S,gamma)
            eigVals, eigVecs = linalg.eigh(S_gamma)
            self.S_QLAM[key] = {'eigVals': eigVals, 'eigVecs': eigVecs}

        inverse = eigVecs.dot(1./(eigVals + c) * np.eye(self.p)).dot(eigVecs.T)

        return inverse
        
    #########################################################################


    #########################################################################
    ###
    ### GET_S_EIGENDECOMPOSITION 
    ### 
    ### Returns QLQ.T decomposition of S_gam with gamma zeroing, not removing
    ###
    ### Last Updated: 4/3/16
    ###

    def getS_QLAM(self,gamma):
        """
            Params: Gamma is the inclusion vector indexing S
            Output: Eigenvalues and Eigenvectors of S_gam increasing order
        """

        if self.check:
            # Verify the type of gamma and that it is the right size and shape
            assert(type(gamma) == np.ndarray)
            assert(gamma.shape == (self.p,1))

        key = str(gamma)
        
        # If we have already computed the eigenvalues, return them from the dictionary
        if self.S_QLAM.has_key(key):
            eigVals = self.S_QLAM.get(key)['eigVals']
            eigVecs = self.S_QLAM.get(key)['eigVecs']
            return (eigVals, eigVecs)

        # Otherwise, find the eigendecomposition and memoize
        else:
            S_gamma = DPPutils.gammaZero2D(self.S,gamma)
            eigVals, eigVecs = linalg.eigh(S_gamma)
            self.S_QLAM[key] = {'eigVals': eigVals, 'eigVecs': eigVecs}
            return (eigVals, eigVecs)
        
    #########################################################################


    #########################################################################
    ###
    ### DET(L)
    ###
    ### Last Updated: 3/31/16
    ###

    def FdetL(self,gamma,theta):
        """
            Params: Theta is the parameterization of L(theta,S)
                    Gamma is the inclusion vector indexing L
            Output: det(L_gamma)
        """

        if self.check:
            # Verify the type, size, and shape of theta and gamma
            assert(type(gamma) == np.ndarray)
            assert(type(theta) == np.ndarray)
            assert(gamma.shape == (self.p,1))
            assert(theta.shape == (self.p,1))

        # If gamma represents the empty set, the determinant is defined as zero.
        if (sum(gamma)[0] < 1.0):
            return 1.0

        key = str(gamma)

        # If we have already computed the eigenvalues, return them from the dictionary
        if self.S_QLAM.has_key(key):
            eigVals = self.S_QLAM.get(key)['eigVals']


        # Otherwise, find the eigendecomposition and memoize
        else:
            S_gamma = DPPutils.gammaZero2D(self.S,gamma)
            eigVals, eigVecs = linalg.eigh(S_gamma)
            self.S_QLAM[key] = {'eigVals': eigVals, 'eigVecs': eigVecs}
            

        
        exponent = sum(gamma * theta)[0]

        detL = np.exp(exponent) * reduce(operator.mul, eigVals[-1 * int(sum(gamma.T[0])):], 1.0)
        # if detL< 1.0e-12:
        #     print "Exponent: %s" % repr(exponent)
        #     print "Gamma: %s" % repr(gamma)
        #     print "theta: %s" % repr(theta)
        #     print "eigVals: %s" % repr(eigVals)
        return detL

    #########################################################################


    #########################################################################
    ###
    ### 1/2 y.T (I - X_gam(X_gam^T X_gam + I)^-1 X_gam^T) y
    ###
    ### Last Updated: 3/16/16
    ###

    def FDifferenceProjection(self,gamma,c):
        """
            Params: Gamma is the inclusion vector indexing S
            Output: 1/2 y.T (X_gam(X_gam^T X_gam + cI)^-1 X_gam^T - I) y
        """

        # if self.check:
        #     assert('gamma' in self.bnv)

        # gamma = self.bnv['gamma'].val_getter()

        if self.check:
            # Verify the type, size, and shape of gamma
            assert(type(gamma) == np.ndarray)
            assert(gamma.shape == (self.p,1))

        key_gamma = str(gamma)
        key_gamma_c = str(gamma) + str(c)

        # If we have already calculated the projection, return it from the dictionary
        if self.differenceProjection.has_key(key_gamma_c):
            return self.differenceProjection.get(key_gamma_c)

        # If gamma is the empty set, inverse is just I * (1/c)
        if (sum(gamma)[0] == 0.0):
            inverse = np.eye(self.p) / c
            eigVecs = np.eye(self.p)

        else:
            # Check if we already have the eigendecomposition
            if self.S_QLAM.has_key(key_gamma):
                eigVals = self.S_QLAM.get(key_gamma)['eigVals']
                eigVecs = self.S_QLAM.get(key_gamma)['eigVecs']
            # Otherwise, find the eigendecomposition and memoize
            else:
                S_gamma = DPPutils.gammaZero2D(self.S,gamma)
                eigVals, eigVecs = linalg.eigh(S_gamma)
                self.S_QLAM[key_gamma] = {'eigVals': eigVals, 'eigVecs': eigVecs}

            # CHECK THAT THIS IS CORRECT
            inverse = np.eye(self.p) / (eigVals + c * np.ones(self.p))

        Xgam = DPPutils.columnGammaZero(self.X,gamma)
        XgamQ = Xgam.dot(eigVecs)
        Diff = np.eye(self.n) - XgamQ.dot(inverse).dot(XgamQ.T)


        # CHECK THIS OUTPUT: * versus .dot() have different outputs for numpy arrays
        # Expect output to be [[value]] -- extract the number.
        projection = 0.5 * self.y.T.dot(Diff).dot(self.y)[0][0]
        self.differenceProjection[key_gamma_c] = projection
        return projection

    #########################################################################



