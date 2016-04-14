#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Variational Inference Framework
####
####  Last updated: 3/20/16
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
mainpath = "/u/tobrund/Engelhardt_DPP"
sys.path.append(os.path.abspath(mainpath))
from Utils.BNV import BNV

import numpy as np
import numpy.matlib as matlib
import scipy.special as funcs
import math as math
import itertools as itertools
import Utils.DPPutils as DPPutils
import Utils.Memoizer as Memoizer


#########################################################################


class VI(object):


    #########################################################################
    ###
    ### INITIALIZER
    ###
    ### Last Updated: 3/16/16
    ###

    def __init__(self, X, y, hp, bnv, **kwargs):

        """
            Params: 
             - X is the n x p matrix of normalized input data 
             - y is the n x 1 column vector of normalized target values
             - hp is a dictionary of all hyperparameter values. Knowledge
                 of naming schemes for use in specifica calculations is
                 expected to be understood by the experimenter, and whoever
                 creates the BNV objects that use them. 
        """
        # Set network parameters
        self.hp = hp
        self.bnv = bnv
        self.X = X
        self.y = y
        self.S = np.transpose(X).dot(X)
        (self.n, self.p) = self.X.shape
        self.b0min = 0.0
        self.verbose = True


        # Set algorithm parameters
        self.max_T = int(1e6) # maximum number of iterations
        self.inner_T = 10 # number of iterations for each variable update per update cycle
        self.check = True
        self.convergenceSitOut = 1e2

        # Set logging defaults
        self.logging = True
        self.dir = 'Data/VI/%s/' % datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')

        for name,value in kwargs.items():
            if name == 'max_T':
                self.max_T = int(value)
            elif name == 'inner_T':
                self.inner_T = int(value)
            elif name == 'b0min':
                self.b0min = float(value)
            elif name == 'check':
                self.check = value
            elif name == 'dir':
                self.dir = value
            elif name == 'logging':
                self.logging = value
            elif name == 'convergenceSitOut':
                self.convergenceSitOut = value
            elif name == 'verbose':
                self.verbose = value

        # Set logging information
        self.logfiles = {}
        if self.logging:
            if not os.path.exists(self.dir):
                try:
                    os.makedirs(self.dir)
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            # Create all log files and note learning rates at the top of each
            for key in self.bnv.keys():
                self.logfiles[key] = '%s/%s.txt' % (self.dir, key)
                if self.bnv[key].isiterative:
                    with open(self.logfiles[key],'a') as f:
                        f.write('Logfile for %s with learning rate %f:\n' % (key, self.bnv[key].alpha))
                else:
                    with open(self.logfiles[key],'a') as f:
                        f.write('Logfile for %s, non-iterative:\n' % key)
            # Log settings
            settingsLog = '%s/%s.txt' % (self.dir, 'settings')
            with open(settingsLog,'w') as f:
                f.write('Data Size: %s\n' % repr(X.shape))
                f.write('Max Iterations: %s\n'% repr(self.max_T))
                f.write('Inner Loop Iterations: %s\n' % repr(self.inner_T))
                f.write('Convergence Sit Out: %s\n' % repr(self.convergenceSitOut))


        if self.check:
            assert(type(X) == np.ndarray)
            assert(type(y) == np.ndarray)
            assert(X.shape[0] == y.shape[0])
            assert(type(hp) == dict)
            assert(type(bnv) == dict)
            for key in bnv:
                assert(issubclass(type(bnv[key]),BNV))

        self.memoizer = Memoizer.Memoizer(self.X, self.y, check=self.check)

        

    #########################################################################
   

    #########################################################################
    ###
    ### VARIATIONAL_INFERENCE
    ###
    ### Last Updated: 4/6/16
    ###

    def variationalInference(self):
        for step in range(self.max_T):
            if self.verbose and step % 100 == 0:
                print "We're on step: %d" % step
                print self.bnv['gamma'].val_getter()
                if self.logging:
                    for fn in self.logfiles.values():
                        with open(fn,'a') as f:
                            f.write('\n>>>>>>>>>\nSTEP %d \n' % step)
            converged = True
            for var in self.bnv.values():
                if var.isiterative:
                    # If it has converged, update sit-out count
                    if var.isconverged():
                        if var.skipped >= self.convergenceSitOut:
                            var.clearHistory()
                        else:
                            var.skipped += 1
                    else:
                        converged = False
                        for t in range(self.inner_T):
                            var.update(self)
                else:
                    var.update(self)
            if converged:
                break



    #########################################################################



    #########################################################################
    ###
    ### PREDICT
    ###
    ### Last Updated: 4/6/16
    ###

    def predict(self, X_test):
        beta = self.getBeta()
        predictions = beta.T.dot(X_test.T).T
        return predictions


    #########################################################################


    #########################################################################
    ###
    ### PREDICT
    ###
    ### Last Updated: 4/7/16
    ###

    def getBeta(self):
        p = self.p
        if 'c' in self.bnv:
            c = self.bnv['c'].val_getter()
        elif 'c' in self.hp:
            c = self.hp['c']
        else:
            print "WARNING - NO VALUE FOUND FOR c. USING DEFAULT c = 1."
            c = 1.0
        gamma = self.bnv['gamma'].val_getter()
        Xgam = DPPutils.columnGammaZero(self.X,gamma)
        inv = np.linalg.inv(c * np.eye(p)+Xgam.T.dot(Xgam))
        beta = self.y.T.dot(Xgam).dot(inv).T
        return beta


    #########################################################################



    ###############
    ## Utilities ##
    ###############

    #########################################################################
    ###
    ### GETL
    ###
    ### Last Updated: 3/17/16
    ###

    def getL(self):
        """
            Params: None - Assume exp(theta/2)X^TXexp(theta/2) form of L
            Output: L
        """
        if self.check:
            assert('theta' in self.bnv)
        theta = self.bnv['theta'].val_getter()

        if self.check:
            # Verify the type of theta and that it is the right size and shape
            assert(type(theta) == np.ndarray)
            assert(theta.shape == (self.p,1))
        
        expTheta = np.exp(0.5 * theta)    # exp(theta/2)
        coeffs = expTheta.dot(expTheta.T) # outer product of exp(theta/2) with itself
        return coeffs * self.S            # pointwise product of coefficients and S (X^T*X)
        
    #########################################################################
