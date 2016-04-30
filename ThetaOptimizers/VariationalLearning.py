#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Variational Learning Theta Optimizer
####
####  Last updated: 4/25/16
####
####  Notes and disclaimers:
####    - I implement Theta as an instance variable, rather than a BNV
####          given the optimizations that are possible when L and gradL
####          are computed simulatneously. 
####    - Gamma is selected via sampling
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
from copy import deepcopy as dc
import time
import datetime
import operator as operator
from pathlib import Path
import warnings

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


class VL(object):

    #########################################################################
    ###
    ### INITIALIZER
    ###
    ### Last Updated: 4/25/16
    ###

    def __init__(self,*argv,**kwargs):
        ######################
        ## Warm Initializer ##
        ###################### 
        if len(argv) == 1:
            VL = argv[0]
            self.X = dc(VL.X)
            self.y = dc(VL.y)
            self.S = dc(VL.S)
            self.n = dc(VL.n)
            self.p = dc(VL.p)
            
            self.max_T = int(5e2)      # Total number of iterations
            self.alpha = dc(VL.alpha)  # Step size
            self.kappa = dc(VL.kappa)

            self.check = dc(VL.check)
            self.verbose = dc(VL.verbose)
            self.logging = dc(VL.logging)
            self.dir = '%swarm_%s/' % (Path(VL.dir).parent,datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S'))

            for name,value in kwargs.items():
                if name == 'max_T':
                    self.max_T = int(value)
                elif name == 'kappa':
                    self.kappa = int(value)
                elif name == 'alpha':
                    self.alpha = float(value)
                elif name == 'check':
                    self.check = value
                elif name == 'verbose':
                    self.verbose = value
                elif name == 'logging':
                    self.logging = value
                elif name == 'dir':
                    self.dir = value

            if self.check:
                assert(type(self.X) == np.ndarray)
                assert(type(self.y) == np.ndarray)
                assert(self.y.shape == (self.X.shape[0],1))


            if self.logging:
                if not os.path.exists(self.dir):
                    try:
                        os.makedirs(self.dir)
                    except OSError as exc: # Guard against race condition
                        raise
                self.thetaFN = "%stheta.txt" % self.dir
                # Log settings
                settingsLog = '%s/%s.txt' % (self.dir, 'settings')
                with open(settingsLog,'a') as f:
                    f.write('Data Size: %s\n' % repr(self.X.shape))
                    f.write('Max Iterations: %s\n'% repr(self.max_T))
                    f.write('Step size: %s\n' % repr(self.alpha))
                    f.write('Kappa from before: %s\n' % repr(VL.kappa))

            self.run_T = self.max_T - VL.max_T

            self.memoizer = dc(VL.memoizer)

            self.c = dc(VL.c)
            self.var = dc(VL.var)

            t0 = time.time()

            self.Cs = dc(VL.Cs)
            self.gs = dc(VL.gs)
            
            self.theta = dc(VL.contTheta)
            L = DPPutils.makeL(self.S,self.theta)
            self.optTheta(VL.contC,VL.contg,L)

            t1 = time.time()

            self.time = t1 - t0

            if self.logging:
                with open(settingsLog,'a') as f:
                    f.write('Time: %s\n' % repr(self.time))

        #####################
        ## New Initializer ##
        ##################### 
        elif len(argv) == 2:
            X = argv[0]
            y = argv[1]
            self.X = X
            self.y = y
            self.S = np.transpose(X).dot(X)
            (self.n, self.p) = self.X.shape
            if self.n <= 2.:
                raise BNV.argumentError("Not enough data: n = %d" % self.n)

            self.max_T = int(1e2)   # Number of iterations
            self.kappa = int(self.p*0.1) # Expected cardinality of gamma
            self.alpha = 1.e-4      # Step size

            self.check = True
            self.verbose = True
            self.logging = True
            self.dir = 'Logs/%s/' % datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')

            for name,value in kwargs.items():
                if name == 'max_T':
                    self.max_T = int(value)
                elif name == 'kappa':
                    self.kappa = int(value)
                elif name == 'alpha':
                    self.alpha = float(value)
                elif name == 'check':
                    self.check = value
                elif name == 'verbose':
                    self.verbose = value
                elif name == 'logging':
                    self.logging = value
                elif name == 'dir':
                    self.dir = value

            if self.check:
                assert(type(X) == np.ndarray)
                assert(type(y) == np.ndarray)
                assert(y.shape == (X.shape[0],1))


            if self.logging:
                if not os.path.exists(self.dir):
                    try:
                        os.makedirs(self.dir)
                    except OSError as exc: # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                self.thetaFN = "%stheta.txt" % self.dir
                # Log settings
                settingsLog = '%s/%s.txt' % (self.dir, 'settings')
                with open(settingsLog,'a') as f:
                    f.write('Data Size: %s\n' % repr(X.shape))
                    f.write('Max Iterations: %s\n'% repr(self.max_T))
                    f.write('Step size: %s\n' % repr(self.alpha))
                    f.write('Kappa: %s\n' % repr(self.kappa))


            self.run_T = self.max_T

            self.memoizer = Memoizer.Memoizer(self.X, self.y, check=self.check)

            self.c = ExperimentUtils.cRROpt(self.X,self.y)

            diffProj = self.memoizer.FDifferenceProjection(np.ones((self.p,1)),self.c)
            self.var = 2. * diffProj / (self.n - 2.)

            t0 = time.time()

            self.Cs = []
            self.gs = []
            if self.kappa < 1:
                self.theta = np.zeros((self.p,1))
            else:
                eVals,_ = self.memoizer.getS_QLAM(np.ones((self.p,1)))
                self.theta = ExperimentUtils.initTheta(eVals, self.kappa) * np.ones((self.p,1))

            L = DPPutils.makeL(self.S,self.theta)
            C = np.diag(DPPutils.getKDiag(L).T[0])
            g = C.dot(self.theta)
            self.optTheta(C,g,L)

            t1 = time.time()

            self.time = t1 - t0

            if self.logging:
                with open(settingsLog,'a') as f:
                    f.write('Time: %s\n' % repr(self.time))

        else:
            print "Usage: __init__(self,X,y,**kwargs) OR __init__(self,ThetaOptimizer,**kwargs)"
    #########################################################################


    #########################################################################
    ###
    ### OPT_THETA
    ###
    ### Last updated: 4/24/16
    ###
    ### Note: Optimize thet value of theta according to Bornn's variational
    ###       learning algorithm. 
    ###

    def optTheta(self, C, g,L):
        """
         Params: theta0 is the starting value of theta
         Result: optimized DPP parameterization, theta 
        """
        if self.check:
            assert self.theta.shape == (self.p, 1)

        for step in range(self.run_T):
            if self.verbose and step % 10 == 0:
                print "Theta Optimization, step: %d " % step
                # print "   %s" % repr(self.theta)
                

            eigVals, eigVecs = linalg.eigh(L)
            gam = ExperimentUtils.DPPSampler(eigVals, eigVecs)

            ghat = self.logPGammaConY(gam, L) * gam  # LINE 6 NORMALIZE THIS
            Chat = DPPutils.getK(L)                  # LINE 7

            try:
                self.gs.append(ghat)
                self.Cs.append(Chat)

                gOld = g
                COld = C 
                g = (1. - self.alpha) * g + self.alpha * ghat
                C = (1. - self.alpha) * C + self.alpha * Chat

                self.theta = linalg.inv(C).dot(g)
                L = DPPutils.makeL(self.S,self.theta)
            except Warning: 
                self.gs = self.gs[:-1]
                self.Cs = self.Cs[:-1]
                g = gOld
                C = COld
                pass

        # Use the last half of ghats and Chats to compute theta
        self.contTheta = self.theta # Saved for continued training
        self.contg = g
        self.contC = C
        Cbar0 = np.zeros(self.Cs[0].shape)
        gbar0 = np.zeros(self.gs[0].shape)
        Navg  = len(self.gs) // 2
        Cbar  = reduce(operator.add,self.Cs[Navg:],Cbar0) 
        gbar  = reduce(operator.add,self.gs[Navg:],gbar0) 
        self.theta = linalg.inv(Cbar).dot(gbar)
        # self.theta = linalg.inv(C).dot(g)
        if self.logging:
            with open(self.thetaFN,'a') as f:
                f.write('%s\n\n' % repr(self.theta))


    #########################################################################


    #########################################################################
    ###
    ### LOG_P_GAMMA_CONDITIONAL_Y
    ###
    ### Last updated: 4/24/16
    ###
    ###

    def logPGammaConY(self,gamma,L):
        det = self.memoizer.FdetSLam(gamma, self.c)
        if det == 0.:
            return -1 * sys.float_info.max
        diffProj = self.memoizer.FDifferenceProjection(gamma, self.c)
        var = self.var
        n = self.n

        logPYcondGamma =  -1. * diffProj / var - 0.5 * np.log(det) \
                          + self.p * 0.5 * np.log(self.c) \
                          - self.n * 0.5 * np.log(2. * np.pi * self.var)

        logGamConTheta = np.log(self.memoizer.FdetL(gamma,self.theta))

        normalizer = np.log(linalg.det(L + np.eye(self.p)))

        return logPYcondGamma + logGamConTheta - normalizer


    #########################################################################

