#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Bayesian Network Variable: c, regression coefficient 
####                                      covariance coefficient. 
####
####  Last updated: 4/3/16
####
####  Notes and disclaimers:
####    - c is the scaling of the covariance matrix for the regression
####        coefficients that distinguishes it from just sig^2, the 
####        variance of the noise on y. 
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

import abc
import numpy as np
import math
import operator as operator
from Utils.BNV import BNV
import scipy.special as funcs
import Utils.DPPutils as DPPutils
from Experiments.VI import VI


#########################################################################


class BN_c(BNV):

    @property 
    def isiterative(self):
        return True

    @property 
    def defaultThreshold(self):
        return 1.0e-6

    @property 
    def defaultAlpha(self):
        return 5.0e-2

    def defaultValue(self):
        return 1.0

    def likelihood(self, state):
        if not issubclass(type(state), VI):
            raise StateError('State must be given in terms of a VI object, not %s.' % type(state).__name__)

        reqKeys = ['a0','b0','gamma']
        self.check_BNVs(state,reqKeys)

        a0 = state.bnv['a0'].val_getter()
        b0 = state.bnv['b0'].val_getter()
        c  = state.bnv['c'].val_getter()
        gamma = state.bnv['gamma'].val_getter()
        p = state.p
        n = state.n

        diffProj = state.memoizer.FDifferenceProjection(gamma,c)

        L = (0.5 * p ** 2 + p) * np.log(c) \
            - 0.5 * np.log(state.memoizer.FdetSLam(gamma,c)) \
            - (a0 + 0.5 * n) * np.log(b0 + diffProj)

        return L

    def gradLikelihood(self, state):
        if not issubclass(type(state), VI):
            raise StateError('State must be given in terms of a VI object, not %s.' % type(state).__name__)

        reqKeys = ['a0','b0','gamma']
        self.check_BNVs(state,reqKeys)

        a0 = state.bnv['a0'].val_getter()
        b0 = state.bnv['b0'].val_getter()
        c  = state.bnv['c'].val_getter()
        gamma = state.bnv['gamma'].val_getter()
        p = state.p
        n = state.n
        y = state.y
        Xgam = DPPutils.columnGammaZero(state.X,gamma)

        eigVals, eigVecs = state.memoizer.getS_QLAM(gamma)
        diffProj = state.memoizer.FDifferenceProjection(gamma,c)

        constant = p / (2.0 * c)#p * (p + 0.5) / c
        sum = 0.5 * reduce(operator.add, \
                           map(lambda x: 1.0 / (x + c), eigVals), \
                           0.0)
        altLam = np.eye(p) / (eigVals + c) ** 2.0
        num = 0.5 * y.T.dot(Xgam) \
                       .dot(eigVecs) \
                       .dot(altLam) \
                       .dot(eigVecs.T) \
                       .dot(Xgam.T) \
                       .dot(y)[0][0]
        denom = b0 + diffProj

        gradL = constant - sum - (a0 + n * 0.5) * num / denom

        return gradL


    def update(self, state):
        grad = self.gradLikelihood(state)
        self.val_setter(state, self.val_getter() + self.alpha * grad)
        if state.logging:
            with open(state.logfiles['c'],'a') as f:
                f.write(' + %f\n%f\n' % (grad,self.val_getter()))
        self.updateChanges(abs(grad))
        return


    def check(self, state, val):
        # Verify that this is enough.
        return val >= 0.0


BNV.register(BN_c)