#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Bayesian Network Variable: b0, Inverse Gamma Scale parameter
####
####  Last updated: 4/2/16
####
####  Notes and disclaimers:
####    - b0 is the scale parameter to the Inverse Gamma prior that we
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
from Utils.BNV import BNV
import numpy as np
import scipy.special as funcs
import scipy.linalg as linalg
import Utils.DPPutils as DPPutils
from Experiments.VI import VI


#########################################################################



class BN_b0(BNV):

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
        default = 1.0
        if len(args) < 1:
            a=1
            #print 'Warning: b0 initialized with default = %f' % default
        elif len(args) == 1:
            default = float(args[0])
        elif len(args) >= 1:
            #print 'Warning: %d unused arguments to b0 initialization.' % (len(args) - 1)
            default = float(args[0])

        return max(default,.01)

    
    def likelihood(self, state):
        # State must be VI object -- we expect it to have memoized
        # Determinant and Inverse lookup functions. 
        if not issubclass(type(state), VI):
            raise StateError('State must be given in terms of a VI object, not %s.' % type(state).__name__)

        # Expected bnvs are a0 and gamma.
        reqKeys = ['a0','gamma','c']
        self.check_BNVs(state,reqKeys)

        gamma = state.bnv['gamma'].val_getter()
        a0 = state.bnv['a0'].val_getter()
        b0 = self.val_getter()
        c  = state.bnv['c'].val_getter()
        n  = state.n

        diffProj = state.memoizer.FDifferenceProjection(gamma,c)

        L = (a0 + 1.0) * np.log(b0) - (a0 + n * 0.5) * np.log(b0 + diffProj)

        return L


    def gradLikelihood(self, state):
        # State must be VI object -- we expect it to have memoized
        # Determinant and Inverse lookup functions. 
        if not issubclass(type(state), VI):
            raise StateError('State must be given in terms of a VI object, not %s.' % type(state).__name__)

        reqKeys = ['a0','gamma']
        self.check_BNVs(state,reqKeys)

        gamma = state.bnv['gamma'].val_getter()
        a0 = state.bnv['a0'].val_getter()
        b0 = self.val_getter()
        c  = state.bnv['c'].val_getter()
        n  = state.n

        diffProj = state.memoizer.FDifferenceProjection(gamma,c)

        gradL = ((a0 + 1.0) / b0) - ((a0 + n * 0.5) / (b0 + diffProj))

        return gradL

    def update(self, state):
        grad = self.gradLikelihood(state)
        self.val_setter(state, self.val_getter() + self.alpha * grad)
        if state.logging:
            with open(state.logfiles['b0'],'a') as f:
                f.write(' + %f\n%f\n' % (grad,self.val_getter()))
        self.updateChanges(abs(grad))
        return

    def check(self, state, val):
        if val >= state.b0min and val > 0.0:
            return True
        Xgam = DPPutils.columnGammaZero(state.X,state.bnv['gamma'].val_getter())
        inverse = np.linalg.inv(Xgam.T.dot(Xgam) + state.bnv['c'].val_getter() * np.eye(state.p))
        M = Xgam.dot(inverse).dot(Xgam.T)

        lam = linalg.eigvalsh(M, eigvals=(state.n - 1, state.n - 1), type=2, overwrite_a=True)[0]
        bound = 0.5 * (lam - 1.0) * state.y.T.dot(state.y)[0][0]
        # print 'bound = %f'%bound
        return val >= bound and val > 0.0

BNV.register(BN_b0)


if __name__ == '__main__':
    asd = BN_b0()
    print issubclass(type(asd), BNV)
