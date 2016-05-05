#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Partial Marginalization Theta Optimizer with LDPP
####
####  Last updated: 4/30/16
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
from copy import deepcopy as dc
import time
import datetime
from pathlib import Path

mainpath = "/Users/Ted/__Engelhardt/Engelhardt_DPP"
sys.path.append(os.path.abspath(mainpath))

import numpy as np
import scipy.linalg as linalg
import itertools as itertools
import DataGeneration.CollinearDataGenerator as CDG
import Utils.ExperimentUtils as ExperimentUtils
import Utils.DPPutils as DPPutils
import Utils.Memoizer as Memoizer
from Utils.BNV import BNV

from sklearn.linear_model import lars_path


#########################################################################

class PM_LDPP(object):

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
            PM = argv[0]
            self.X = dc(PM.X)
            self.y = dc(PM.y)
            self.S = dc(PM.S)
            self.n = dc(PM.n)
            self.p = dc(PM.p)
            
            self.max_T = int(5e2)               # Total number of iterations to optimize theta
            self.alphaTheta = dc(PM.alphaTheta) # Learning rate on optimization of Theta
            self.alphaW = dc(PM.alphaW)         # Learning rate on optimization of w
            self.tau = dc(PM.tau)               # Tolerance

            self.ignore = dc(PM.ignore)

            self.check = dc(PM.check)
            self.verbose = dc(PM.verbose)
            self.logging = dc(PM.logging)
            self.dir = '%s/warm_%s/' % (Path(PM.dir).parent,datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S'))

            for name, value in kwargs.items():
                if name == 'max_T':
                    self.max_T = int(value)
                elif name == 'alphaTheta':
                    self.alphaTheta = float(value)
                elif name == 'alphaW':
                    self.alphaW = float(value)
                elif name == 'tau':
                    self.tau = float(value)
                elif name == 'kappa':
                    self.kappa = int(value)
                elif name == 'check':
                    self.check = value
                elif name == 'verbose':
                    self.verbose = value
                elif name == 'logging':
                    self.logging = value
                elif name == 'dir':
                    self.dir = value

            self.run_T = self.max_T - PM.max_T # Number of extra iterations needed

            self.memoizer = dc(PM.memoizer)

            self.c   = dc(PM.c)
            self.var = dc(PM.var)

            self.larsTopGamma = dc(PM.larsTopGamma)
            self.larsCap = dc(PM.larsCap)


            if self.check:
                assert(type(self.X) == np.ndarray)
                assert(type(self.y) == np.ndarray)
                assert(self.y.shape == (self.X.shape[0],1))

            if self.logging:
                if not os.path.exists(self.dir):
                    try:
                        os.makedirs(self.dir)
                    except OSError as exc: # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                self.thetaFN = "%stheta.txt" % self.dir
                self.wFN = "%sw.txt" % self.dir

                # Log settings
                settingsLog = '%s/%s.txt' % (self.dir, 'settings')
                with open(settingsLog,'a') as f:
                    f.write('Data Size: %s\n' % repr(self.X.shape))
                    f.write('Max Iterations: %s\n'% repr(self.max_T))
                    f.write('Learning Rate Theta: %s\n' % repr(self.alphaTheta))
                    f.write('Learning Rate W: %s\n' % repr(self.alphaW))
                    f.write('Num Ignored @ Start: %s\n' % repr(len(self.ignore)))
                    f.write('Lars Cap @ Start: %s\n' % repr(self.larsCap))


            t0 = time.time()

            self.theta = dc(PM.theta)
            self.w = dc(PM.w)
            self.gradientAscent()

            t1 = time.time()

            self.time = t1 - t0

            if self.logging:
                with open(settingsLog,'a') as f:
                    f.write('Time: %s\n' % repr(self.time))
                    f.write('Num Ignored @ End: %s\n' % repr(len(self.ignore)))
                    f.write('Lars Cap @ End: %s\n' % repr(self.larsCap))

        #####################
        ## New Initializer ##
        #####################         
        elif len(argv) == 2:
            X = argv[0]
            y = argv[1]
            self.X = X
            self.y = y
            self.S = X.T.dot(X)
            (self.n, self.p) = self.X.shape
            if self.n <= 2.:
                raise BNV.ArgumentError("Not enough data: n = %d" % self.n)
            
            self.max_T = int(5e2)   # Number of iterations to optimize theta
            self.alphaTheta = 1. # Learning rate on optimization of Theta
            self.alphaW = 1.e-1     # Learning rate on optimization of w
            self.tau = 1.e-4        # Tolerance
            self.larsCap = 10       # Minimum number of features to marginalize over

            self.check = True
            self.verbose = True
            self.logging = True
            self.dir = 'Logs/%s/' % datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')

            for name, value in kwargs.items():
                if name == 'max_T':
                    self.max_T = int(value)
                elif name == 'alphaTheta':
                    self.alphaTheta = float(value)
                elif name == 'alphaW':
                    self.alphaW = float(value)
                elif name == 'tau':
                    self.tau = float(value)
                elif name == 'larsCap':
                    self.larsCap = int(value)
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

            # Set number of iterations to run, and build memoizer
            self.run_T = self.max_T
            self.memoizer = Memoizer.Memoizer(self.X, self.y, check=self.check)

            # Solve for c and sig^2
            self.c = ExperimentUtils.cRROpt(self.X,self.y)
            diffProj = self.memoizer.FDifferenceProjection(np.ones((self.p,1)),self.c)
            self.var = 2. * diffProj / (self.n - 2.)

            # Create Search and Ignore Sets
            self.larsTopGamma = self.larsSet()
            self.ignore = set()

            if self.logging:
                if not os.path.exists(self.dir):
                    try:
                        os.makedirs(self.dir)
                    except OSError as exc: # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                self.thetaFN = "%stheta.txt" % self.dir
                self.wFN = "%sw.txt" % self.dir
                # Log settings
                settingsLog = '%s%s.txt' % (self.dir, 'settings')
                with open(settingsLog,'a') as f:
                    f.write('Data Size: %s\n' % repr(X.shape))
                    f.write('Max Iterations: %s\n'% repr(self.max_T))
                    f.write('Learning Rate Theta: %s\n' % repr(self.alphaTheta))
                    f.write('Learning Rate W: %s\n' % repr(self.alphaW))
                    f.write('Num Ignored @ Start: %s\n' % repr(len(self.ignore)))


            t0 = time.time()

            self.theta = 10
            self.w = 1
            self.gradientAscent()

            t1 = time.time()

            self.time = t1 - t0

            if self.logging:
                with open(settingsLog,'a') as f:
                    f.write('Time: %s\n' % repr(self.time))
                    f.write('Num Ignored @ End: %s\n' % repr(len(self.ignore)))

        else:
            print "Usage: __init__(self,X,y,**kwargs) OR __init__(self,ThetaOptimizer,**kwargs)"
    #########################################################################



    #########################################################################
    ###
    ### THETA_OPTIMIZER
    ###
    ### Last Updated: 4/26/16
    ###
    ### Note: Returns the optimal value for theta marginalized over gammas 
    ###       generated from the LARS set.
    ###

    def gradientAscent(self):
        thetaConverge = False
        wConverge = False

        for step in range(self.run_T):
            if thetaConverge and wConverge:
                break
            if self.verbose:
                sys.stdout.write('\r%d' % (self.max_T - self.run_T + step))
                sys.stdout.flush()

            L, gradLTheta, gradLW = self.theta_L_gradL_gamma(self.theta, self.w)
            if not thetaConverge:
                newTheta = self.theta + self.alphaTheta * gradLTheta
                self.theta = max(0., newTheta)
            if not wConverge:
                self.w += self.alphaW * gradLW

            if self.logging:
                if not thetaConverge:
                    with open(self.thetaFN,'a') as f:
                        f.write('<<<<<<<<<<< STEP %d >>>>>>>>>>>>\n' % step)
                        f.write(' + %s\n%s\n' % (repr(gradLTheta),repr(self.theta)))
                if not wConverge:
                    with open(self.wFN,'a') as f:
                        f.write('<<<<<<<<<<< STEP %d >>>>>>>>>>>>\n' % step)
                        f.write(' + %s\n%s\n' % (repr(gradLW),repr(self.w)))

            # If we're under the threshold, log it. Otherwise,
            # restart count.
            if abs(gradLTheta) < self.tau:
                thetaConverge = True
            if abs(gradLW) < self.tau: 
                wConverge = True



    #########################################################################



    #########################################################################
    ###
    ### LARS_SET
    ###
    ### Last Updated: 4/20/16
    ###
    ### Note: Returns the first features that capture proportion p of the 
    ###       "correlation" in LARS. These are not necessarily the variables
    ###       with the largest coefficients. They are taken in the order that
    ###       they were selected by LARS.
    ###

    def larsSet(self):#,p=.75,cap=10):
        alpha, order, coefs = lars_path(self.X,self.y.T[0])
        return order

    #########################################################################


    #########################################################################
    ###
    ### THETA_LIKELIHOOD_AND_GRADIENT
    ###
    ### Last Updated: 4/24/16
    ###
    ### Note: Returns the likelihood and its gradient for a given value of theta.
    ###

    def theta_L_gradL_gamma(self,theta,w):

        if theta == 0.:
            return 0., 0.5/self.alphaTheta, 0.

        if w == 0.:
            return 0., 0., 1.e3


        gammaLamFactor = np.exp(theta) - 1.
        pLamFactor = np.exp(theta) + np.exp(theta - w) - 1.

        normEigVals, _ = self.memoizer.getS_QLAM(np.ones((self.p,1)))
        normalizer = np.exp(self.p * (w - theta)) \
                     * reduce(lambda a, x: a * (x + pLamFactor), normEigVals, 1.0)

        def getGamma(idx):
            gamma = np.zeros((self.p,1))
            for i in idx:
                gamma[i,0] = 1.
            return gamma

        L = 0.
        GradLTheta = 0.
        GradLW = 0.
        nothing = True
        for i in range(self.larsCap):
            for p in itertools.combinations(self.larsTopGamma[0:self.larsCap],i+1):
                gamma = getGamma(p)

                if p in self.ignore:
                    continue
                else:
                    nothing = False

                logPYGam = self.logPYconditionalGammaX(gamma)
                
                detLGam = self.memoizer.LDPP_FdetL(gamma,theta,w)

                # Compute coefficient in the logspace
                k = np.exp(logPYGam + np.log(detLGam) - np.log(normalizer))

                if k == 0.:
                    # If it's too small, and p(y|gamma) was negligable on its own, 
                    if np.exp(logPYGam) == 0.:
                        self.ignore.add(p)

                L += k

                gamEigVals, _ = self.memoizer.getS_QLAM(gamma)
                d = int(sum(gamma)[0])
                
                gammaSum = reduce(lambda a, x: a + 1./(x + gammaLamFactor),gamEigVals[-1 * d:],0.)
                pSum     = reduce(lambda a, x: a + 1./(x + pLamFactor), normEigVals, 0.)

                GradLTheta += k * ((gammaLamFactor + 1.) * gammaSum 
                                    - (pLamFactor + 1.) * pSum 
                                    + self.p - d)
                GradLW += k * (d - self.p + np.exp(theta - w) * pSum)

                # print "\n\n\n\n"
                # print gamEigVals
                # print gamEigVals[-1 * sum(gamma)[0]:]
                # print gamma
                # print k
                # print k * ((sum(gamma)[0] - self.p) / theta - gradSum1 + gradSum2)
                # print gradSum1
                # print gradSum2
                # print pGamTheta
                # print normalizer
                # print logPYGam
        if nothing:
            print "HOLD UP - EVERYTHING IS IGNORED"
            self.larsCap += 1

        # if L == 0.:
        #     GradLogL = GradL * 1e6  # Don't divide by zero, but jump!
        # #     L = -1 * sys.float_info.max
        # else:
        #     GradLogL = GradL / L    # Otherwise, compute Gradient of Log (bigger)
        #     L = np.log(L)
        # if type(L) == np.ndarray:
        #     print L

        if type(GradLTheta) == np.ndarray:
            print GradLTheta
        if type(GradLW) == np.ndarray:
            print GradLW
        return L, GradLTheta/L, GradLW/L



    #########################################################################


    #########################################################################
    ###
    ### LOG_P_Y_CONDITIONAL_GAMMA_X
    ###
    ### Last Updated: 4/24/16
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




