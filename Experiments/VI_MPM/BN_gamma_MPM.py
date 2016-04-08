#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Bayesian Network Variable: Gamma, the inclusion vector.
####        For use in the Maximally Pre-Marginalized Network
####
####  Last updated: 4/2/16
####
####  Notes and disclaimers:
####    - Gamma is the binary inclusion vector from {0,1}^p, drawn from
####          a DPP prior, parameterized on theta.
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


class BN_gamma(BNV):

    @property 
    def isiterative(self):
        return False

    @property 
    def defaultThreshold(self):
        return 'Should never get here'

    @property 
    def defaultAlpha(self):
        return 'Should never get here'

    def defaultValue(self, *args):
        defaultP = 1

        if len(args) < 1:
            a=0
            #print 'Warning: Gamma initialized with default p = %d' % defaultP
        elif len(args) == 1:
            defaultP = int(args[0])
        elif len(args) >= 1:
            #print 'Warning: %d unused arguments to Theta initialization' % (len(args) - 1)
            defaultP = int(args[0])

        # Use random binary vector with at least one non-zero element
        # gam = np.vstack((np.array([1]),\
        #         np.array([np.random.binomial(1,0.5,defaultP - 1)]).T))
        gam = np.ones((defaultP,1))
        return gam

    def likelihood(self, state):
        # State must be VI object -- we expect it to have memoized
        # Determinant and Inverse lookup functions. 
        if not issubclass(type(state), VI):
            raise StateError('State must be given in terms of a VI object, not %s.' % type(state).__name__)

        reqKeys = ['theta','a0','b0','c']
        self.check_BNVs(state,reqKeys)
        reqParams = ['lam_gamma']
        self.check_HPs(state,reqParams)

        theta = state.bnv['theta'].val_getter()
        a0    = state.bnv['a0'].val_getter()
        b0    = state.bnv['b0'].val_getter()
        c     = state.bnv['c'].val_getter()
        gamma = self.val_getter()
        p = state.p
        n = state.n
        lam_gamma = state.hp['lam_gamma']

        inclusionSum = sum(gamma * theta)[0] # ndarray point product is *
        diffProj = state.memoizer.FDifferenceProjection(gamma,c)

        # print state.memoizer.FdetL(gamma,np.zeros((p,1)))
        # print "FdetSLam: %s" % repr(state.memoizer.FdetSLam(gamma,c))
        # print "gam: %s" % repr(gamma)
        # print "c: %f" % c
        
        L = inclusionSum + np.log(state.memoizer.FdetL(gamma,np.zeros((p,1)))) \
            - 0.5 * np.log(state.memoizer.FdetSLam(gamma,c)) \
            - (a0 + 0.5 * n) * np.log(b0 + diffProj) \
            - lam_gamma * sum(gamma)[0]
        # print L
        return L

    def gradLikelihood(self, state):
        return 'Should Never Get Here'

    def update(self, state):
        if not issubclass(type(state), VI):
            raise StateError('State must be given in terms of a VI object, not %s.' % type(state).__name__)

        reqKeys = ['a0','b0']
        self.check_BNVs(state,reqKeys)
        reqParams = ['lam_gamma']
        self.check_HPs(state,reqParams)

        # a0    = state.bnv['a0'].val_getter()
        # b0    = state.bnv['b0'].val_getter()
        # theta = state.bnv['theta'].val_getter()
        p = state.p
        # lam_gamma = state.hp['lam_gamma']

        L = lambda gam: self.dummy(state,gam)

        gamma = DPPutils.greedyMapEstimate(p,L)
        self.val_setter(state,gamma)

        # with open('temp.txt','a') as f:
        #     f.write('\ngamma: %s\n'% repr(self.val_getter()))
        #     f.write('theta: %s\n'%repr(state.bnv['theta'].val_getter()))
        #     f.write('a0: %f\n'%state.bnv['a0'].val_getter())
        #     f.write('b0: %f\n'%state.bnv['b0'].val_getter())
        if state.logging:
            with open(state.logfiles['gamma'],'a') as f:
                f.write('\n%s\n' % repr(self.val_getter()))
        return

    def check(self, state, val):
        if type(val) == np.ndarray:
            return val.shape == (state.p,1) \
                and np.logical_or(val == 1.0, val == 0.0).all()
        else:
            return False



    def dummy(self,state,gam):
        self.val_setter(state,gam)
        return self.likelihood(state)


BNV.register(BN_gamma)


