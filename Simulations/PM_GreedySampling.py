#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: PM Greedy and Sampling Predictor Generator
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

mainpath = "/u/tobrund/Engelhardt_DPP"
sys.path.append(os.path.abspath(mainpath))

import numpy as np
import pickle
import Utils.DPPutils as DPPutils
import ThetaOptimizers.PartialMarginalization as PM
import GammaSelectors.Greedy as Greedy
import GammaSelectors.Sampling as Sampling
import Predictor as Predictor


#########################################################################

setStart = int(sys.argv[1])
setFinal = int(sys.argv[2])

dataFolders = ['n_025','n_050','n_075','n_100','n_150','n_200','n_400']
Tthetas = [10,20,50,100,200,500]

for i in range(setStart,setFinal):
    setDir = 'Set%02d/' % i
    for df in dataFolders:
        currentDir = '%s%s/' % (setDir,df)
        X_tr = np.load('%sX_tr.npy' % currentDir)
        y_tr = np.load('%sy_tr.npy' % currentDir)

        for T in Tthetas:
            try:
                logDir = '%sPM_Theta_%03d/' % (currentDir,T)
                if T == Tthetas[0]:
                    PM_theta = PM.PM(X_tr,y_tr,max_T=T,verbose=False,dir=logDir)
                else:
                    PM_theta = PM.PM(PM_theta,max_T=T,verbose=False,dir=logDir)
            except: 
                break
            ###########################
            ## Make Greedy Predictor ##
            ###########################
            greedy_gamma = Greedy.Greedy(PM_theta)
            greedy_predictor = Predictor.Predictor(X_tr,y_tr,greedy_gamma.gamma,c=greedy_gamma.c)
            pickle.dump(greedy_predictor, open('%sPM_Theta%03d_Greedy.p' % (logDir,T),'wb'))

            #############################
            ## Make Sampling Predictor ##
            #############################
            sampling_gamma = Sampling.Sampling(PM_theta)
            sampling_predictor = Predictor.Predictor(X_tr,y_tr,sampling_gamma.gamma, c=sampling_gamma.c)
            pickle.dump(sampling_predictor, open('%sPM_Theta%03d_Sampling.p' % (logDir,T),'wb'))

