#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Basic Analysis
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
import dill
import pickle

mainpath = "/Users/Ted/__Engelhardt/Engelhardt_DPP"
sys.path.append(os.path.abspath(mainpath))

import numpy as np
import scipy.linalg as linalg
import scipy.io as IO
import PredictorWrapper as PredictorWrapper
import Utils.DPPutils as DPPutils


#########################################################################



setStart = int(sys.argv[1])
setFinal = int(sys.argv[2])

dataFolders = ['n_025','n_050','n_075','n_100','n_150','n_200','n_400']
Tthetas = [10,20,50,100,200,500]

for i in range(setStart,setFinal):
    setDir = 'Set%02d/' % i
    betaStar = np.load('%sbetaStar.npy' % setDir)
    gammaStar = np.load('%sgammaStar.npy' % setDir)


    N = float(setFinal - setStart)

    # mse, beta, sparsity, over, under

    def betaLoss(predictor):
        loss = max(abs(predictor.beta.T[0] - betaStar.T[0]))
        return loss

    def excessFeatures(predictor):
        excess = sum([1. if predictor.gamma[i,0] > gammaStar[i,0] else 0. for i in range(predictor.gamma.shape[0])])
        return excess

    def missedFeatures(predictor):
        missed = sum([1. if gammaStar[i,0] > predictor.gamma[i,0] else 0. for i in range(predictor.gamma.shape[0])])
        return missed

    def sparsity(predictor):
        sprs = float(sum(predictor.gamma)[0]) / predictor.gamma.shape[0]
        return sprs

    measurements = {'mse': None, 
                    'beta': betaLoss, 
                    'sprs': sparsity,
                    'over': excessFeatures,
                    'undr': missedFeatures}

    # Measurements for NON SPARSE regressions
    NSmeasurements = ['mse','beta']
    def update(dictionary, predictor, i, j):
        for k in dictionary.keys():
            # print k
            # if k == 'mse':
                # print dictionary[k]
            # print dictionary[k][i,j]
            dictionary[k][i,j] += measurements[k](predictor) / N
            # print dictionary[k][i,j]
    
    def empty(keys,shape):
        d = {}
        for key in keys:
            d[key] = np.zeros(shape)
        return d

    OLSR  = empty(NSmeasurements,(1,len(dataFolders)))
    RIDGE = empty(NSmeasurements,(1,len(dataFolders)))
    LARS2 = empty(measurements.keys(),(1,len(dataFolders)))
    LARS4 = empty(measurements.keys(),(1,len(dataFolders)))
    LARS10= empty(measurements.keys(),(1,len(dataFolders)))
    LARS  = empty(measurements.keys(),(1,len(dataFolders)))
    LASSO = empty(measurements.keys(),(1,len(dataFolders)))
    SRVM  = empty(measurements.keys(),(1,len(dataFolders)))
    PMG   = empty(measurements.keys(),(len(Tthetas),len(dataFolders)))
    PMS   = empty(measurements.keys(),(len(Tthetas),len(dataFolders)))



    for j,df in enumerate(dataFolders):
        currentDir = '%s%s/' % (setDir,df)
        X_te = np.load('%sX_te.npy' % currentDir)
        y_te = np.load('%sy_te.npy' % currentDir)

        def mse(predictor):
            yhat = np.reshape(predictor.predict(X_te),y_te.shape)
            error = sum((yhat - y_te) ** 2)[0]
            return error

        measurements['mse'] = mse


        lasso = dill.load(open('%sStandardRegressions/LASSO.p' % currentDir,'rb'))
        update(LASSO,lasso,0,j)

        olsr = dill.load(open('%sStandardRegressions/OLSR.p' % currentDir, 'rb'))
        update(OLSR,olsr,0,j)

        ridge = dill.load(open('%sStandardRegressions/RIDGE.p' % currentDir, 'rb'))
        update(RIDGE,ridge,0,j)

        lars2 = dill.load(open('%sStandardRegressions/LARS_02.p' % currentDir,'rb'))
        update(LARS2,lars2,0,j)

        lars4 = dill.load(open('%sStandardRegressions/LARS_04.p' % currentDir,'rb'))
        update(LARS4,lars4,0,j)

        lars10 = dill.load(open('%sStandardRegressions/LARS_10.p' % currentDir,'rb'))
        update(LARS10,lars10,0,j)

        lars = dill.load(open('%sStandardRegressions/LARS_OPT.p' % currentDir,'rb'))
        update(LARS,lars,0,j)

        SRVMinfo = IO.loadmat('%ssRVM.mat' % currentDir)
        P = lambda X: X.dot(SRVMinfo['beta'])
        srvm = PredictorWrapper.PredictorWrapper(SRVMinfo['beta'],SRVMinfo['gamma'],P)
        update(SRVM,srvm,0,j)


        for k,T in enumerate(Tthetas):
            thetaG = dill.load(open('%sPM_Theta_%03d/PM_Theta%03d_Greedy.p' % (currentDir,T,T),'rb'))
            update(PMG,thetaG,k,j)

            thetaS = dill.load(open('%sPM_Theta_%03d/PM_Theta%03d_Sampling.p' % (currentDir,T,T),'rb'))
            update(PMS,thetaS,k,j)


    
def saveAll(dictionary,fname):
        for k in dictionary.keys():
            np.save('%s_%s' % (fname,k),dictionary[k],allow_pickle=True)

saveAll(LASSO,'Averages/lasso')
saveAll(OLSR,'Averages/olsr')
saveAll(RIDGE,'Averages/ridge')
saveAll(LARS2,'Averages/lars2')
saveAll(LARS4,'Averages/lars4')
saveAll(LARS10,'Averages/lars10')
saveAll(LARS,'Averages/lars')
saveAll(SRVM,'Averages/srvm')
saveAll(PMG,'Averages/PMG')
saveAll(PMS,'Averages/PMS')



