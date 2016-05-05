#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Greedy Gamma Selector
####
####  Last updated: 5/4/16
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
# from Experiments.PO_PG import PO
import Utils.DPPutils as DPPutils


#########################################################################


class LDPP_Greedy(object):

    #########################################################################
    ###
    ### INITIALIZER
    ###
    ### Last Updated: 4/25/16
    ###

    def __init__(self,TO,lam_gamma=0.0):
        self.theta = dc(TO.theta)
        self.w = dc(TO.w)
        self.var = dc(TO.var)
        self.c = dc(TO.c)
        self.p = dc(TO.p)
        self.n = dc(TO.n)
        self.lam_gamma = lam_gamma
        self.memoizer = dc(TO.memoizer)

        def likelihood(gamma):
            diffProj = self.memoizer.FDifferenceProjection(gamma,self.c)
            L = np.log(self.memoizer.LDPP_FdetL(gamma,self.theta,self.w)) \
                - 0.5 * np.log(self.memoizer.FdetSLam(gamma,self.c)) \
                - diffProj / self.var \
                - self.lam_gamma * sum(gamma)[0]
            return L

        L = lambda gam: likelihood(gam)

        t0 = time.time()

        self.gamma = DPPutils.greedyMapEstimate(self.p,L)

        t1 = time.time()

        self.time = t1 - t0

        if TO.logging:
            settingsLog = '%s/%s.txt' % (TO.dir, 'GreedySettings')
            with open(settingsLog,'a') as f:
                f.write('Lambda_gamma: %s\n' % repr(self.lam_gamma))
                f.write('Time: %s\n' % repr(self.time))

    #########################################################################

