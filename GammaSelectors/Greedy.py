#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Greedy Gamma Selector
####
####  Last updated: 4/24/16
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

import numpy as np
import scipy.linalg as linalg
# from Experiments.PO_PG import PO
import Utils.DPPutils as DPPutils


#########################################################################


class Greedy(object):

    #########################################################################
    ###
    ### INITIALIZER
    ###
    ### Last Updated: 4/24/16
    ###

    def __init__(self,thetaOptimizer,lam_gamma=0.0):