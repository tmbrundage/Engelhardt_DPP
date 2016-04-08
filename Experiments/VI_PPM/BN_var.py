#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Bayesian Network Variable: var, Variance of labels
####
####  Last updated: 4/3/16
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
mainpath = "/u/tobrund/Engelhardt_DPP"
sys.path.append(os.path.abspath(mainpath))

import abc
from Utils.BNV import BNV
import numpy as np
import Utils.DPPutils as DPPutils
from Experiments.VI import VI

#########################################################################

class BN_var(BNV):

    @property 
    def isiterative(self):
        return True

    @property
    def defaultAlpha(self):
        return 1.0e-3

    @property 
    def defaultThreshold(self):
        return 1.0e-6

    def defaultValue(self):
        return 1.0

    def likelihood(self, state):
        if not issubclass(type(state), VI):
            raise StateError('State must be given in terms of a VI object, not %s.' % type(state).__name__)

        reqKeys = ['gamma']
        self.check_BNVs(state,reqKeys)
        reqParams = ['a0','b0','c']
        self.check_HPs(state,reqParams)

        gamma = state.bnv['gamma'].val_getter()
        a0 = state.hp['a0']
        b0 = state.hp['b0']
        c  = state.hp['c']
        var = self.val_getter()
        n = state.n

        diffProj = state.memoizer.FDifferenceProjection(gamma,c)

        L = -1.0 * diffProj / var - (a0 + 1.0 + n * 0.5) * np.log(var)

        return L

    def gradLikelihood(self, state):
        if not issubclass(type(state), VI):
            raise StateError('State must be given in terms of a VI object, not %s.' % type(state).__name__)

        reqKeys = ['gamma']
        self.check_BNVs(state,reqKeys)
        reqParams = ['a0','b0','c']
        self.check_HPs(state,reqParams)

        gamma = state.bnv['gamma'].val_getter()
        a0 = state.hp['a0']
        b0 = state.hp['b0']
        c  = state.hp['c']
        var = self.val_getter()
        n = state.n

        diffProj = state.memoizer.FDifferenceProjection(gamma,c)

        gradL = diffProj / (var ** 2.0) - (a0 + 1.0 + n * 0.5) / var

        return gradL
        

    def update(self, state):
        grad = self.gradLikelihood(state)
        self.val_setter(state, self.val_getter() + self.alpha * grad)
        if state.logging:
            # >>>>> CHECK THE LOGGING OUTPUT
            with open(state.logfiles['var'],'a') as f:
                f.write(' + %s\n%s\n' % (repr(grad),repr(self.val_getter())))
        self.updateChanges(abs(grad))
        return

    def check(self, state, val):
        return val > 0.0

BNV.register(BN_var)