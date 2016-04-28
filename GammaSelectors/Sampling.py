#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Sampling Gamma Selector
####
####  Last updated: 4/25/16
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
from copy import deepcopy as dc
import datetime

mainpath = "/Users/Ted/__Engelhardt/Engelhardt_DPP"
sys.path.append(os.path.abspath(mainpath))

import numpy as np
import numpy.matlib as matlib
import scipy.special as funcs
import scipy.linalg as linalg
import math as math
import random as random
import itertools as itertools
import DataGeneration.CollinearDataGenerator as CDG
import Utils.ExperimentUtils as ExperimentUtils
import Utils.DPPutils as DPPutils
import Utils.Memoizer as Memoizer
from Utils.BNV import BNV


#########################################################################


class Sampling(object):

    #########################################################################
    ###
    ### INITIALIZER
    ###
    ### Last Updated: 4/25/16
    ###

    def __init__(self,TO,**kwargs):
        self.theta = dc(TO.theta)
        self.X = dc(TO.X)
        self.y = dc(TO.y)
        self.S = dc(TO.S)
        self.n = dc(TO.n)
        self.p = dc(TO.p)
        self.c = dc(TO.c)
        self.var = dc(TO.var)

        self.max_T = int(5e2)    # Number of samples

        self.check = dc(TO.check)
        self.verbose = dc(TO.verbose)
        self.logging = dc(TO.logging)
        self.dir = dc(TO.dir)

        for name,value in kwargs.items():
            if name == 'max_T':
                self.max_T = int(value)
            elif name == 'check':
                self.check = value
            elif name == 'verbose':
                self.verbose = value
            elif name == 'logging':
                self.logging = value
            elif name == 'dir':
                self.dir = value

        if self.check:
            assert(type(self.theta) == np.ndarray)
            assert(self.theta.shape == (self.p,1))

        if self.logging:
            if not os.path.exists(self.dir):
                try:
                    os.makedirs(self.dir)
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            settingsLog = '%s/%s.txt' % (TO.dir, 'SamplingSettings')
            with open(settingsLog,'a') as f:
                f.write('Number of samples: %s\n' % repr(self.max_T))

        self.memoizer = dc(TO.memoizer)

        t0 = time.time()

        self.gamma = self.gammaSamplingSelector()

        t1 = time.time()

        self.time = t1 - t0

        if self.logging:
            with open(settingsLog,'a') as f:
                f.write('Time: %s\n' % repr(self.time))


    #########################################################################


    #########################################################################
    ###
    ### GAMMA_SAMPLING_SELECTOR
    ###
    ### Last Updated: 4/25/16
    ###
    ### Note: Returns the optimal value for theta marginalized over gammas 
    ###       generated from the LARS set.
    ###

    def gammaSamplingSelector(self):
        
        # Build the L ensemble with self's theta. Find the eigendecomposition
        L = DPPutils.makeL(self.S,self.theta)

        eigVals, eigVecs = linalg.eigh(L)

        maxGamma = np.zeros((self.p,1))
        maxLikelihood = -1 * sys.float_info.max#self.gammaLikelihood(maxGamma)

        tested = set()
        tested.add(repr(maxGamma))

        for sample in range(self.max_T):
            gammaTest = ExperimentUtils.DPPSampler(eigVals, eigVecs)
            if repr(gammaTest) not in tested:
                likelihood = self.gammaLikelihood(gammaTest)
                if likelihood > maxLikelihood:
                    maxLikelihood = likelihood
                    maxGamma = gammaTest
                tested.add(repr(gammaTest))

        return maxGamma


    #########################################################################




    #########################################################################
    ###
    ### GAMMA_LIKELIHOOD
    ###
    ### Last Updated: 4/25/16
    ###
    ###

    def gammaLikelihood(self, gamma):
        # No need to include normalization of p(gamma|theta)
        logPYGam = self.logPYconditionalGammaX(gamma)
        pGamTheta = self.memoizer.FdetL(gamma,self.theta)
        if pGamTheta == 0. or sum(gamma)[0] == 0.:
            return -1 * sys.float_info.max
        return logPYGam + np.log(pGamTheta)


    #########################################################################



    #########################################################################
    ###
    ### LOG_P_Y_CONDITIONAL_GAMMA_X
    ###
    ### Last Updated: 4/25/16
    ###
    
    def logPYconditionalGammaX(self,gamma):
        # p(y|gamma)
        det = self.memoizer.FdetSLam(gamma, self.c)
        if det == 0.:
            return -1 * sys.float_info.max
        diffProj = self.memoizer.FDifferenceProjection(gamma,self.c)
        var = self.var
        n = self.n

        logPYcondGamma = -1. * diffProj / var - 0.5 * np.log(det)
        
        return logPYcondGamma

    #########################################################################

