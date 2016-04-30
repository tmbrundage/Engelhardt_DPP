#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Predictor Wrapper
####
####  Last updated: 4/29/16
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


class PredictorWrapper(object):

    #########################################################################
    ###
    ### INITIALIZER
    ###
    ### Last Updated: 4/29/16
    ###

    def __init__(self, beta, gamma, P):
        self.beta = dc(beta)
        self.gamma = dc(gamma)
        self.predict = P

    #########################################################################



