#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Parameter Optimizer
####
####  Last updated: 4/13/16
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
import time
import datetime
mainpath = "/Users/Ted/__Engelhardt/Engelhardt_DPP"
sys.path.append(os.path.abspath(mainpath))
from Utils.BNV import BNV

import numpy as np
import numpy.matlib as matlib
import scipy.special as funcs
import math as math
import itertools as itertools
import Utils.DPPutils as DPPutils
import Utils.Memoizer as Memoizer

from sklearn.linear_model import lars_path


#########################################################################





class PO(object):


    #########################################################################
    ###
    ### INITIALIZER
    ###
    ### Last Updated: 4/13/16
    ###


    def __init__(self,X,y,**kwargs):
        self.X = X
        self.y = y
        self.S = np.transpose(X).dot(X)
        (self.n, self.p) = self.X.shape
        if self.n <= 2.:
            raise BNV.ArgumentError("Not enough data: n = %d" % self.n)
        
        self.max_T = 1e5
        self.alpha = 1.e-2
        self.tau = 1.e-4
        self.GA_max_T = 1e5

        self.check = True
        self.verbose = True
        self.logging = True
        self.dir = 'Logs/%s/' % datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')

        a0 = 100.

        for name,value in kwargs.items():
            if name == 'max_T':
                self.max_T = int(value)
            elif name == 'dir':
                self.dir = value
            elif name == 'logging':
                self.logging = value
            elif name == 'check':
                self.check = value
            elif name == 'a0':
                a0 = float(value)
            elif name == 'alpha':
                self.alpha = float(value)
            elif name == 'tau':
                self.tau = float(tau)
            elif name == 'GA_max_T':
                self.GA_max_T = 1e5

        if self.check:
            assert(type(X) == np.ndarray)
            assert(type(y) == np.ndarray)
            assert(y.shape == (X.shape[0],1))

        self.memoizer = Memoizer.Memoizer(self.X, self.y, check=self.check)


        c = self.cRROpt()
        lam_min = np.log10(opt_c)-1.
        lam_max = lam_min + 2.
        c = self.cRROpt(lam_min=lam_min,lam_max=lam_max)

        diffProj = self.memoizer.FDifferenceProjection(np.ones((self.p,1)),c)
        b0 = (a0 + 1.) * diffProj / (self.n - 2.)
        hp = {}
        hp['a0'] = a0
        hp['b0'] = b0
        hp['c']  = c

        larsSet = self.larsSet(p=0.75,cap=10)

    #########################################################################


    #########################################################################
    ###
    ### THETA_OPTIMIZER
    ###
    ### Last Updated: 4/13/16
    ###
    ### Note: Returns the optimal value for theta marginalized over gammas 
    ###       generated from the LARS set.
    ###

    def gradientAscentTheta(self):


    #########################################################################


    #########################################################################
    ###
    ### C_RR_OPT
    ###
    ### Last Updated: 4/13/16
    ###
    ### Note: Returns the optimal value for the regularization parameter in
    ###       ridge regression. Note - we're not splitting into a train and
    ###       validation set yet.
    ###

    def cRROpt(self, n=20,lam_min=-5,lam_max=15):
        # DOES IT MAKE A DIFFERENCE IF I DO THIS WITH ALL TRAINING DATA
        # OR DO I NEED TO FORCE A VALIDATION SET?

        def Eval(learned):
            learned_yhat = self.X.dot(learned)
            learned_mse = sum((self.y - learned_yhat) ** 2)
            return learned_mse

        def Learn(lam):
            inverse = self.memoizer.FinvSLam(np.ones((self.p,1)))
            learned_beta = inverse.dot(self.X.T).dot(y)
            return learned_beta

        lams = np.logspace(lam_min,lam_max,n)

        opt_c = ExperimentUtils.gridSearch1D(lams,Learn,Eval)

        return opt_c


    #########################################################################



    #########################################################################
    ###
    ### LARS_SET
    ###
    ### Last Updated: 4/13/16
    ###
    ### Note: Returns the first features that capture proportion p of the 
    ###       "correlation" in LARS. These are not necessarily the variables
    ###       with the largest coefficients. They are taken in the order that
    ###       they were selected by LARS.
    ###

    def larsSet(self,p=.75,cap=10):
        alphas, order, coefs = lars_path(X,y.T[0])

        magnitudes = np.array([abs(coefs[i,-1]) for i in order])
        total = sum(quantities)

        partialSums = np.array(reduce(lambda a, x: a + [a[-1] + x], magnitudes[1:], [magnitudes[0]]))

        maxIdx = 1
        while partialSums[maxIdx - 1] < p:
            maxIdx += 1

        return order[0:min(maxIdx, cap)]

    #########################################################################




    #########################################################################
    ###
    ### THETA_LIKELIHOOD_AND_GRADIENT
    ###
    ### Last Updated: 4/13/16
    ###
    ### Note: Returns the likelihood and its gradient for a given value of theta.
    ###

    def theta_L_gradL_gamma(self,theta):

        # Calculate all values that are constant for every term in the sum
        expTheta = np.exp(theta * 0.5) * np.eye(self.p)
        L = expTheta.dot(self.S).dot(expTheta) + np.eye(self.p)
        eigVals, eigVecs = np.linalg.eigh(L)
        normalizer = reduce(lambda a, x: a * (x + 1.), eigVals, 1.)
        K = np.eye(self.p) - eigVecs.dot((1./(eigVals+1.)) * np.eye(self.p)).dot(eigVecs.T)
        K1 = K.dot(np.ones((self.p,1)))

        def getGamma(idx):
            gamma = np.zeros((self.p,1))
            for i in idx:
                gamma[i,0] = 1.
            return gamma

        L = 0.
        GradL = 0.
        for i in range(len(self.larsSet)):
            for p in itertools.combinations(self.larsSet,i+1):
                gamma = getGama(p)
                pYGam = self.pYconditionalGammaX(gamma,theta)
                pGamTheta = self.memoizer.FdetL(gamma,theta)
                k = pYGam * pGamTheta
                L += k
                GradL += k * (gamma - K1)

        L *= normalizer
        GradL *= normalizer

        return accL, accGradL

        """ RECURSIVE DEFINITION 
        # def aux(accL, accGradL, gamma, set):
        #     pYGam = self.pYconditionalGammaX(gamma,theta)
        #     pGamTheta = self.memoizer.FdetL(gamma,theta)
        #     k = pYGam * pGamTheta
        #     L = accL + k
        #     GradL = accGradL + k * (gamma - K1)
        #     if len(set) == 1:
        #         return accL, accGradL
        #     else:
        #         e = set[-1]
        #         e_next = set[-2]
        #         set_next = set[:-1]
        #         gamma1 = gamma
        #         gamma1[e_next:0] = 1.
        #         # Call with next element set to 1.
        #         aux(L, GradL, gamma1, set_next)
        #         gamma0 = gamma1
        #         gamma0[e:0] = 0.
        #         # Call with this element reset to 0 and next element set to 1. 
        #         aux(L, GradL, gamma0, set_next)

        # gamma = np.zeros((self.p,1))
        # # P(gamma = 0_p) = 0, since det(L_0) == 0
        # accL = 0.
        # accGradL = np.zeros((self.p,1))
        # set = self.larsSet
        # gamma[set[-1]:0] = 1.
        # accL, accGradL = aux(accL, accGradL, gamma, set)
        """


    #########################################################################


    #########################################################################
    ###
    ### P_Y_CONDITIONAL_GAMMA_X
    ###
    ### Last Updated: 4/13/16
    ###
    
    def pYconditionalGammaX(self,gamma,theta):
        # p(y|gamma)
        det = self.memoizer.FdetSLam(gamma, self.hp['c'])
        diffProj = self.memoizer.FDifferenceProjection(gamma,self.hp['c'])
        b0 = self.hp['b0']
        a0 = self.hp['a0']
        n = self.n
        pYcondGamma = 1.0 / (np.sqrt(det) * (b0 + diffProj) ** (a0 + n * 0.5))

        return pYcondGamma

    #########################################################################



