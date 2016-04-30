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

mainpath = "/Users/Ted/__Engelhardt/Engelhardt_DPP"
sys.path.append(os.path.abspath(mainpath))

import numpy as np
import pickle
import Utils.DPPutils as DPPutils
import ThetaOptimizers.VariationalLearning as VL
import GammaSelectors.Greedy as Greedy
import GammaSelectors.Sampling as Sampling
import Predictor as Predictor
import warnings


#########################################################################

setStart = int(sys.argv[1])
setFinal = int(sys.argv[2])

dataFolders = ['n_025','n_050','n_075','n_100','n_150','n_200','n_400']
Tthetas = [1000,5000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,25000]
Kappas = [2,4,10]

for i in range(setStart,setFinal):
    setDir = 'Set%02d/' % i
    for df in dataFolders:
        currentDir = '%s%s/' % (setDir,df)
        X_tr = np.load('%sX_tr.npy' % currentDir)
        y_tr = np.load('%sy_tr.npy' % currentDir)

        for kappa in Kappas:
            for T in Tthetas:
                try:
                    logDir = '%sVL_Kappa_%02d_Theta_%03d/' % (currentDir,kappa,T)
                    if T == Tthetas[0]:
                        VL_theta = VL.VL(X_tr,y_tr,max_T=T,kappa=kappa,verbose=False,dir=logDir)
                    else:
                        VL_theta = VL.VL(VL_theta,max_T=T,kappa=kappa,verbose=False,dir=logDir)
                except: 
                    break

                ###########################
                ## Make Greedy Predictor ##
                ###########################
                
                #############################
                ## Make Sampling Predictor ##
                #############################
                try:
                    sampling_gamma = Sampling.Sampling(VL_theta)
                    sampling_predictor = Predictor.Predictor(X_tr,y_tr,sampling_gamma.gamma, c=sampling_gamma.c)
                    pickle.dump(sampling_predictor, open('%sVL_Kappa%02d_Theta%03d_Sampling.p' % (logDir,kappa,T),'wb'))
                except:
                    greedy_gamma = Greedy.Greedy(VL_theta)
                    greedy_predictor = Predictor.Predictor(X_tr,y_tr,greedy_gamma.gamma,c=greedy_gamma.c)
                    pickle.dump(greedy_predictor, open('%sVL_Kappa%02d_Theta%03d_Greedy.p' % (logDir,kappa,T),'wb'))

