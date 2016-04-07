#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Experiment Utilitiex
####
####  Last updated: 4/7/16
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
mainpath = "/Users/Ted/__Engelhardt/Code"
sys.path.append(os.path.abspath(mainpath))
import time
import datetime

import numpy as np
import scipy.linalg as linalg

from Experiments.VI import VI

#########################################################################


#########################################################################
###
### GRID_SEARCH_1D
###
### Last Updated: 4/7/16
###

def gridSearch1D(grid, Learn, Eval, MAX = False, verbose = False):

    if MAX:
        best = sys.float_info.min 
    else:
        best = sys.float_info.max 

    bestVal = grid[0]

    for var in grid:
        L = Learn(var)
        current = Eval(L)
        while type(current) == np.ndarray:
            current = current[0]
        if MAX:
            if current > best:
                best = current
                bestVal = var
        else:
            if current < best:
                best = current
                bestVal = var

        if verbose:
            print "Current Loss: %f   for Lambda: %f" % (current, var)

    return bestVal


#########################################################################

