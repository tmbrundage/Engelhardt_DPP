#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: PM Greedy and Sampling Predictor Generator
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
# import pickle
import dill
import Predictor as Predictor
import PredictorWrapper as PredictorWrapper
from sklearn.linear_model import Ridge, Lasso, Lars
import Utils.ExperimentUtils as ExperimentUtils


#########################################################################


setStart = int(sys.argv[1])
setFinal = int(sys.argv[2])

dataFolders = ['n_025','n_050','n_075','n_100','n_150','n_200','n_400']

for i in range(setStart,setFinal):
    setDir = 'Set%02d/' % i
    for df in dataFolders:
        currentDir = '%s%s/' % (setDir,df)
        X_tr = np.load('%sX_tr.npy' % currentDir)
        y_tr = np.load('%sy_tr.npy' % currentDir)

        val_size = int(0.1 * X_tr.shape[0])
        X_val = X_tr[0:val_size,:]
        y_val = y_tr[0:val_size,:]
        X_train = X_tr[val_size:,:]
        y_train = y_tr[val_size:,:]

        logDir = '%sStandardRegressions/' % currentDir
        if not os.path.exists(logDir):
            os.makedirs(logDir)
        logFile = '%sLogs.txt' % logDir

        ##########
        ## OLSR ##
        ##########
        olsr_predictor = Predictor.Predictor(X_tr,y_tr,gamma=np.ones((X_tr.shape[1],1)))
        dill.dump(olsr_predictor,open('%sOLSR.p' % logDir,'wb'))

        ###########
        ## RIDGE ##
        ###########
        ridgeLams = np.logspace(-5,6,500)

        def ridgeEval(learned):
            learned_yhat = learned.predict(X_val)
            learned_mse = sum((y_val - learned_yhat) ** 2)[0]
            return learned_mse

        def ridgeLearn(lam):
            ridge = Ridge(alpha=lam,fit_intercept=False,copy_X=True)
            ridge.fit(X_train,y_train)
            return ridge

        optLam = ExperimentUtils.gridSearch1D(ridgeLams, ridgeLearn, ridgeEval, MAX=False)
        ridge_predictor = Predictor.Predictor(X_tr,y_tr,gamma=np.ones((X_tr.shape[1],1)),c=optLam)
        dill.dump(ridge_predictor,open('%sRIDGE.p' % logDir,'wb'))
        with open(logFile,'a') as f:
            f.write('Ridge c: %15.10f\n' % optLam)

        ###########
        ## LASSO ##
        ###########
        lassoLams = np.logspace(-5,6,500)

        def lassoEval(learned):
            learned_yhat = learned.predict(X_val)
            learned_mse = sum((y_val - learned_yhat) ** 2)[0]
            return learned_mse

        def lassoLearn(lam):
            lasso = Lasso(alpha=lam,fit_intercept=False,copy_X=True,max_iter=1.e7,tol=.0001)
            lasso.fit(X_train,y_train)
            return lasso

        optLam = ExperimentUtils.gridSearch1D(lassoLams, lassoLearn, lassoEval, MAX=False)
        lasso = Lasso(alpha=optLam,fit_intercept=False,copy_X=True,max_iter=1.e7,tol=.0001)
        lasso.fit(X_tr,y_tr)
        lasso_beta = np.array([lasso.coef_]).T
        lasso_gamma = np.array([[0. if abs(x) < 1e-100 else 1. for x in lasso.coef_]]).T
        # P = lambda X: lasso.predict(X)
        lasso_predictor = PredictorWrapper.PredictorWrapper(lasso_beta,lasso_gamma,lasso.predict)
        dill.dump(lasso_predictor,open('%sLASSO.p' % logDir,'wb'))
        with open(logFile,'a') as f:
            f.write('Lasso c: %15.10f        alpha: %15.10f\n' % (1./(2.* X_tr.shape[0]), optLam))



        ##############
        ## LARS_SET ##
        ##############
        kappa = [2,4,10]
        for k in kappa:
            lars = Lars(n_nonzero_coefs=k,fit_intercept=False)
            lars.fit(X_tr,y_tr)
            lars_beta = np.array([lars.coef_]).T
            lars_gamma = np.zeros((X_tr.shape[1],1))
            lars_gamma[lars.active_] = 1.
            lars_predictor = PredictorWrapper.PredictorWrapper(lars_beta,lars_gamma,lars.predict)
            dill.dump(lars_predictor,open('%sLARS_%02d.p' % (logDir,k),'wb'))

        ##############
        ## LARS_OPT ##
        ##############
        larsKappas = np.linspace(0,40,41,dtype=int)

        def larsEval(learned):
            learned_yhat = learned.predict(X_val)
            learned_mse = sum((y_val - learned_yhat) ** 2)[0]
            return learned_mse

        def larsLearn(kap):
            lars = Lars(n_nonzero_coefs=kap,fit_intercept=False)
            lars.fit(X_train,y_train)
            return lars

        optKap = ExperimentUtils.gridSearch1D(larsKappas,larsLearn, larsEval, MAX=False)
        lars = Lars(n_nonzero_coefs=optKap,fit_intercept=False)
        lars.fit(X_tr,y_tr)
        lars_beta = np.array([lars.coef_]).T
        lars_gamma = np.zeros((X_tr.shape[1],1))
        lars_gamma[lars.active_] = 1.
        lars_predictor = PredictorWrapper.PredictorWrapper(lars_beta,lars_gamma,lars.predict)
        dill.dump(lars_predictor,open('%sLARS_OPT.p' % logDir,'wb'))
        with open(logFile,'a') as f:
            f.write('Lars optimized n_nonzero_coefs: %d \n' % optKap)

