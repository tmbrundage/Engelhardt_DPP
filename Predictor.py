#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Parameter Optimizer
####
####  Last updated: 4/28/16
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
import scipy.linalg as linalg
import Utils.DPPutils as DPPutils


#########################################################################


class Predictor(object):

    #########################################################################
    ###
    ### INITIALIZER
    ###
    ### Last Updated: 4/25/16
    ###

    def __init__(self, X, y, gamma, c=0):
        self.gamma = dc(gamma)
        Xgam = DPPutils.columnGammaZero(X, gamma)
        Minv = linalg.inv(c * np.eye(X.shape[1]) + Xgam.T.dot(Xgam))
        self.beta = dc(y.T.dot(Xgam).dot(Minv).T)


    #########################################################################


    #########################################################################
    ###
    ### PREDICT
    ###
    ### Last Updated: 4/25/16
    ###

    def predict(self, X_test):
        beta = self.beta
        predictions = X_test.dot(beta)
        return predictions


    #########################################################################


