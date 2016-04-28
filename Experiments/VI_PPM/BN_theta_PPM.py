#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Bayesian Network Variable: Theta, the parameterization of L 
####        For use in the Maximally Pre-Marginalized Network
####
####  Last updated: 4/2/16
####
####  Notes and disclaimers:
####    - Theta is the parameterization of the L ensemble, used to define 
####          the DPP prior on gamma, the inclusion vector. As discussed
####          in my notes, the parameterization is:
####            L = exp(theta/2)(X^T)(X)exp(theta/2)
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
import Utils.theta_priors as theta_priors
from Experiments.VI import VI

#########################################################################


class BN_theta(BNV):

    @property
    def isiterative(self):
        return True

    @property
    def defaultAlpha(self):
        return 1.0e-2

    @property 
    def defaultThreshold(self):
        return 1.0e-6

    def defaultValue(self, *args):
        defaultP = 1
        
        if len(args) < 1:
            a=0
            # print 'Warning: Theta initialized with default p = %d' % defaultP
        elif len(args) == 1:
            defaultP = int(args[0])
        elif len(args) >= 1:
            # print 'Warning: %d unused arguments to Theta initialization' % (len(args) - 1)
            defaultP = int(args[0])

        # Use folded unit normal distribution 
        # expTheta = np.array([abs(np.random.randn(defaultP))]).T
        # theta0 = np.log(expTheta)
        theta0 = np.zeros((defaultP,1))
        return theta0


    def __init__(self, **kwargs):
        super(BN_theta, self).__init__(**kwargs)
        self.logThetaPrior = lambda state: theta_priors.uniformLikelihood(state)
        self.gradLogThetaPrior = lambda state: theta_priors.uniformGradLIkelihood(state)
        for name,value in kwargs.items():
            if name == 'prior':
                if value == 'uniform':
                    self.logThetaPrior = lambda state: theta_priors.uniformLikelihood(state)
                    self.gradLogThetaPrior = lambda state: theta_priors.gradUniformLikelihood(state)
                elif value == 'El0':
                    self.logThetaPrior = lambda state: theta_priors.expectationL0LogLikelihood(state)
                    self.gradLogThetaPrior = lambda state: theta_priors.gradExpectationL0LogLikelihood(state)
                elif value == 'gaussian':
                    self.logThetaPrior = lambda state: theta_priors.gaussianLogLikelihood(state)
                    self.gradLogThetaPrior = lambda state: theta_priors.gradGaussianLogLikelihood(state)


    def likelihood(self, state):
        # State must be VI object -- we expect it to have memoized
        # Determinant and Inverse lookup functions. 
        if not issubclass(type(state), VI):
            raise StateError('State must be given in terms of a VI object, not %s.' % type(state).__name__)

        # Expected bnv is just gamma.
        reqKeys = ['gamma']
        self.check_BNVs(state,reqKeys)

        gamma = state.bnv['gamma'].val_getter()
        theta = self.val_getter()

        inclusionSum = sum(gamma * theta)[0] # ndarray point product is *
        L = inclusionSum - np.log(state.memoizer.FdetL(np.array([np.ones(state.p)]).T,theta)) \
             + self.logThetaPrior(state) 

        return L

    def gradLikelihood(self, state):
        # State must be VI object -- we expect it to have memoized
        # Determinant and Inverse lookup functions. 
        if not issubclass(type(state), VI):
            raise StateError('State must be given in terms of a VI object, not %s.' % type(state).__name__)

        # Expected bnv is just gamma.
        reqKeys = ['gamma']
        self.check_BNVs(state,reqKeys)

        gamma = state.bnv['gamma'].val_getter()
        Kdiag = DPPutils.getKDiag(state.getL())

        gradL = gamma - Kdiag + self.gradLogThetaPrior(state)

        return gradL

    def update(self, state):
        grad = self.gradLikelihood(state)
        self.val_setter(state, self.val_getter() + self.alpha * grad)
        if state.logging:
            with open(state.logfiles['theta'],'a') as f:
                f.write(' + %s\n%s\n' % (repr(grad),repr(self.val_getter())))
        self.updateChanges(max(abs(grad))[0])
        return

    def check(self, state, val):
        # Checks that val is an appropriately shaped NDarray
        if type(val) == np.ndarray:
            return val.shape == (state.p, 1)
        else:
            return False


BNV.register(BN_theta)


