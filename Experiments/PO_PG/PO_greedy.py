#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Parameter Optimizer
####
####  Last updated: 4/16/16
####
####  Notes and disclaimers:
####    - I implement Theta as an instance variable, rather than a BNV
####          given the optimizations that are possible when L and gradL
####          are computed simulatneously. 
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
import time
import datetime

mainpath = "/Users/Ted/__Engelhardt/Engelhardt_DPP"
sys.path.append(os.path.abspath(mainpath))

import numpy as np
import scipy.linalg as linalg
# from Experiments.PO_PG import PO
import Utils.DPPutils as DPPutils


#########################################################################



class PO_greedy(object):

    #########################################################################
    ###
    ### INITIALIZER
    ###
    ### Last Updated: 4/16/16
    ###


    def __init__(self,PO,lam_gamma = 1.0):
        self.X = PO.X
        self.y = PO.y
        self.theta = PO.theta
        self.c = PO.c
        self.var = PO.var
        self.p = PO.p
        self.n = PO.n
        self.lam_gamma = lam_gamma

        def likelihood(gamma):
            inclusionSum = sum(gamma * self.theta)[0]
            diffProj = PO.memoizer.FDifferenceProjection(gamma,self.c)
            L = inclusionSum + np.log(PO.memoizer.FdetL(gamma,np.zeros((self.p,1)))) \
                - 0.5 * np.log(PO.memoizer.FdetSLam(gamma,self.c)) \
                - diffProj / (2.0 * self.var) \
                - self.lam_gamma * sum(gamma)[0]
            return L

        L = lambda gam: likelihood(gam)

        self.gamma = DPPutils.greedyMapEstimate(self.p,L)

        Xgam = DPPutils.columnGammaZero(self.X,self.gamma)
        Minv = linalg.inv(self.c * np.eye(self.p) + Xgam.T.dot(Xgam))
        self.beta = self.y.T.dot(Xgam).dot(Minv).T


    #########################################################################



    #########################################################################
    ###
    ### PREDICT
    ###
    ### Last Updated: 4/16/16
    ###

    def predict(self, X_test):
        beta = self.beta
        predictions = X_test.dot(beta)
        return predictions


    #########################################################################


