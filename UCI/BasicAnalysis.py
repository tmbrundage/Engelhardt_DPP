#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Basic Analysis
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

Tthetas = [10,20,50,100,200,500]
N = setFinal - setStart


def sparsity(predictor):
    sprs = float(sum(predictor.gamma)[0]) / predictor.gamma.shape[0]
    return sprs

measurements = {'mse': None,  
                'sprs': sparsity}

# Measurements for NON SPARSE regressions
NSmeasurements = ['mse']
def update(dictionary, predictor, row, col):
    for key in dictionary.keys():
        dictionary[key][row,col] += measurements[key](predictor)

def empty(keys,shape):
    d = {}
    for key in keys:
        d[key] = np.zeros(shape)
    return d

OLSR  = empty(NSmeasurements,(1,N))
RIDGE = empty(NSmeasurements,(1,N))
LARS2 = empty(measurements.keys(),(1,N))
LARS4 = empty(measurements.keys(),(1,N))
LARS10= empty(measurements.keys(),(1,N))
LARS  = empty(measurements.keys(),(1,N))
LASSO = empty(measurements.keys(),(1,N))
SRVM  = empty(measurements.keys(),(1,N))
PMG   = empty(measurements.keys(),(len(Tthetas),N))
PMS   = empty(measurements.keys(),(len(Tthetas),N))

for i in range(setStart,setFinal):
    setDir = 'Fold%d/' % i
    

    X_te = np.load('%sX_te.npy' % setDir)
    y_te = np.array([np.load('%sy_te.npy' % setDir)]).T

    def mse(predictor):
        yhat = predictor.predict(X_te)
        error = sum((yhat - y_te) ** 2)[0]
        return error

    measurements['mse'] = mse


    lasso = dill.load(open('%sStandardRegressions/LASSO.p' % setDir,'rb'))
    update(LASSO,lasso,0,i)

    olsr = dill.load(open('%sStandardRegressions/OLSR.p' % setDir, 'rb'))
    update(OLSR,olsr,0,i)

    ridge = dill.load(open('%sStandardRegressions/RIDGE.p' % setDir, 'rb'))
    update(RIDGE,ridge,0,i)

    lars2 = dill.load(open('%sStandardRegressions/LARS_02.p' % setDir,'rb'))
    update(LARS2,lars2,0,i)

    lars4 = dill.load(open('%sStandardRegressions/LARS_04.p' % setDir,'rb'))
    update(LARS4,lars4,0,i)

    lars10 = dill.load(open('%sStandardRegressions/LARS_10.p' % setDir,'rb'))
    update(LARS10,lars10,0,i)

    lars = dill.load(open('%sStandardRegressions/LARS_OPT.p' % setDir,'rb'))
    update(LARS,lars,0,i)

    SRVMinfo = IO.loadmat('%ssRVM.mat' % setDir)
    P = lambda X: X.dot(SRVMinfo['beta'])
    srvm = PredictorWrapper.PredictorWrapper(SRVMinfo['beta'],SRVMinfo['gamma'],P)
    update(SRVM,srvm,0,i)


    for k,T in enumerate(Tthetas):
        thetaG = dill.load(open('%sPM_Theta_%03d/PM_Theta%03d_Greedy.p' % (setDir,T,T),'rb'))
        update(PMG,thetaG,k,i)
        if i==0:
            print thetaG.beta
        
        thetaS = dill.load(open('%sPM_Theta_%03d/PM_Theta%03d_Sampling.p' % (setDir,T,T),'rb'))
        update(PMS,thetaS,k,i)


    
def saveAll(dictionary,fname):
        for k in dictionary.keys():
            np.save('%s_%s' % (fname,k),dictionary[k],allow_pickle=True)

saveAll(LASSO,'Summary/lasso')
saveAll(OLSR,'Summary/olsr')
saveAll(RIDGE,'Summary/ridge')
saveAll(LARS2,'Summary/lars2')
saveAll(LARS4,'Summary/lars4')
saveAll(LARS10,'Summary/lars10')
saveAll(LARS,'Summary/lars')
saveAll(SRVM,'Summary/srvm')
saveAll(PMG,'Summary/PMG')
saveAll(PMS,'Summary/PMS')



