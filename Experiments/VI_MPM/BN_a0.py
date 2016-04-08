#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Bayesian Network Variable: a0, Inverse Gamma Shape parameter
####
####  Last updated: 4/2/16
####
####  Notes and disclaimers:
####    - a0 is the shape parameter to the Inverse Gamma prior that we
####        put on the variance. Treating this as a BNV and not a hyper-
####        parameter implies that we have already performed maximal 
####        marginalization, integrating out both the regression 
####        coefficients and the variance. 
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
mainpath = "/u/tobrund/Engelhardt_DPP"
sys.path.append(os.path.abspath(mainpath))

import abc
import numpy as np
import math
from Utils.BNV import BNV
import scipy.special as funcs
import Utils.DPPutils as DPPutils
from Experiments.VI import VI


#########################################################################


class BN_a0(BNV):

    @property
    def isiterative(self):
        return True

    @property 
    def defaultThreshold(self):
        return 1.0e-3

    @property
    def defaultAlpha(self):
        return 1.0e-2

    def defaultValue(self,*args):
        return 1.0

    
    def likelihood(self, state):
        # State must be VI object -- we expect it to have memoized
        # Determinant and Inverse lookup functions. 
        if not issubclass(type(state), VI):
            raise StateError('State must be given in terms of a VI object, not %s.' % type(state).__name__)

        # Expected bayesian network variables include b0 and gamma.
        reqKeys = ['b0','gamma','c']
        self.check_BNVs(state,reqKeys)

        gamma = state.bnv['gamma'].val_getter()
        b0 = state.bnv['b0'].val_getter()
        a0 = self.val_getter()
        c  = state.bnv['c'].val_getter()
        n = state.n

        diffProj = state.memoizer.FDifferenceProjection(gamma,c)

        L = funcs.gammaln(n * 0.5 + a0) - funcs.gammaln(a0) \
             + a0 * (np.log(b0) - np.log(b0 + diffProj)) \
             + 0.5 * np.log(a0 * funcs.polygamma(1,a0) - 1.0)

        return L

    def gradLikelihood(self, state):
        # State must be VI object -- we expect it to have memoized
        # Determinant and Inverse lookup functions. 
        if not issubclass(type(state), VI):
            raise StateError('State must be given in terms of a VI object, not %s.' % type(state).__name__)

        # Expected bayesian network variables include b0 and gamma.
        reqKeys = ['b0','gamma','c']
        self.check_BNVs(state,reqKeys)

        gamma = state.bnv['gamma'].val_getter()
        b0 = state.bnv['b0'].val_getter()
        a0 = self.val_getter()
        c  = state.bnv['c'].val_getter()
        n = state.n
        diffProj = state.memoizer.FDifferenceProjection(gamma,c)
        

        # VERIFY THAT SCIPY USES 0 AS FIRST DERIVATIVE
        gradL = funcs.polygamma(0, n * 0.5 + a0) \
                 - funcs.polygamma(0, a0) \
                 + np.log(b0) - np.log(b0 + diffProj) \
                 + 0.5 * (funcs.polygamma(1,a0) + a0 * funcs.polygamma(2,a0)) \
                 / (a0 * funcs.polygamma(1,a0) - 1.0)

        if math.isnan(gradL):
            print 'NAN IN when a0 = %f'%a0
            print "b0: %f"%b0
            print "diffProj: %f"%diffProj

        return gradL

    def update(self, state):
        grad = self.gradLikelihood(state)
        self.val_setter(state, self.val_getter() + self.alpha * grad)
        self.updateChanges(abs(grad))
        if state.logging:
            with open(state.logfiles['a0'],'a') as f:
                f.write(' + %f\n%f\n' % (grad,self.val_getter()))
        return

    def check(self, state, val):
        # Right now, only constraint on a0 is that it's positive
        return val >= 0.0

BNV.register(BN_a0)


