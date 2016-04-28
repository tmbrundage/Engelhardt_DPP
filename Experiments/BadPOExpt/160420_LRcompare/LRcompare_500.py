#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Linear Regression Comparison Experiment
####
####  Last updated: 4/16/16
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
mainpath = "/u/tobrund/Engelhardt_DPP/"
sys.path.append(os.path.abspath(mainpath))
import time
import datetime

import numpy as np
import scipy.linalg as linalg

from Experiments.PO_PG import PO
from Experiments.PO_PG import PO_greedy

import DataGeneration.CollinearDataGenerator as CDG
import Utils.DPPutils as DPPutils
import Utils.Memoizer as Memoizer
import Utils.ExperimentUtils as ExperimentUtils

from sklearn.linear_model import Ridge, Lasso

#########################################################################

ns = [50,75,100,150,200,300,400]
n_test = 1000
p=40
sparsity=0.90
T = 500
##################
## Housekeeping ##
##################


loggingDirectory = 'Logs_P%d_T%d/%s/' % (p,int(T),datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S'))

def Eval(learned):
    learned_yhat = learned.predict(X_val)
    learned_mse = sum((y_val - learned_yhat) ** 2)
    return learned_mse

outputDir = "%soutput/" % loggingDirectory
if not os.path.exists(outputDir):
    try:
        os.makedirs(outputDir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

def mseFile(expt):
    return "%s%s_mse.txt" % (outputDir,expt)
def betaFile(expt):
    return "%s%s_beta.txt" % (outputDir,expt)
def lamFile(expt):
    return "%s%s_lam.txt" % (outputDir,expt)

repeat = 100

for rep in range(repeat):
    for n in ns:
        ####################
        ## Establish Data ##
        ####################

        print "<<<<< N = %d >>>>>" % n
        dataGen = CDG.CollinearDataGenerator(p=p, sparsity=sparsity)
        X = dataGen.getX(n)
        p = X.shape[1]
        y = dataGen.getY(X)
        # y_mu = np.mean(y)
        # y_range = np.max(y) - np.min(y)
        # y = (y - y_mu) / (y_range)

        X_test = dataGen.getX(n_test)
        y_test = dataGen.getY(X_test)
        # y_test = (y_test - y_mu) / (y_range)

        val_size = int(0.1 * X.shape[0])
        X_val = X[0:val_size,:]
        y_val = y[0:val_size,:]
        X_train = X[val_size:,:]
        y_train = y[val_size:,:]

        oracleBeta = dataGen.betaStar
        
        ########################################
        ## DPP:                               ##
        ##  - Exponential Parameterization    ##
        ##  - Partial gamma Marginalization   ##     
        ##      for theta optimization.       ##
        ##  - Sampling optimization of gamma  ##
        ########################################

        DPP_PO = PO.PO(X,y,max_T = 500, GA_max_T = T)

        DPP_PO_yhat = DPP_PO.predict(X_test)
        DPP_PO_mse = sum((y_test.T[0] - DPP_PO_yhat.T[0]) ** 2)
        DPP_PO_betaLoss = max(abs(DPP_PO.beta - oracleBeta))[0]
        with open(mseFile('DPP_PO'),'a') as f:
            f.write("%15.10f    " % DPP_PO_mse)
        with open(betaFile('DPP_PO'),'a') as f:
            f.write("%15.10f    " % DPP_PO_betaLoss)
        with open(lamFile('DPP_PO'),'a') as f:
            f.write("%15.10f    " % DPP_PO.c)
        print "DPP_PO MSE: %f    BETA_LOSS: %f   OPT_LAM: %f    TOTAL:%d    DROPPED:%d" % (DPP_PO_mse, DPP_PO_betaLoss, DPP_PO.c, sum(DPP_PO.gamma),len(DPP_PO.ignore))

      

        ########################################
        ## DPP:                               ##
        ##  - Exponential Parameterization    ##
        ##  - Greedy MAP estimate of gamma    ##
        ##      given PO estimate of theta    ##     
        ########################################


        lams = [0.,1.e-5,1.e-4,1.e-3,1.e-2,1.e-1,1.,5.,10.,20.]

        def Learn(lam):
            DPP_PO_greedy = PO_greedy.PO_greedy(DPP_PO,lam)
            return DPP_PO_greedy

        def DPP_PO_greedy_eval(learned):
            learned_yhat = learned.predict(X)
            learned_mse = sum((y - learned_yhat) ** 2)
            return learned_mse
    
        
        optLam = ExperimentUtils.gridSearch1D(lams, Learn, DPP_PO_greedy_eval, MAX=False)

        DPP_PO_greedy = PO_greedy.PO_greedy(DPP_PO,optLam)

        DPP_PO_greedy_yhat = DPP_PO_greedy.predict(X_test)
        DPP_PO_greedy_mse = sum((y_test.T[0] - DPP_PO_greedy_yhat.T[0]) ** 2)
        DPP_PO_greedy_betaLoss = max(abs(DPP_PO_greedy.beta - oracleBeta))[0]
        with open(mseFile('DPP_PO_greedy'),'a') as f:
            f.write("%15.10f    " % DPP_PO_greedy_mse)
        with open(betaFile('DPP_PO_greedy'),'a') as f:
            f.write("%15.10f    " % DPP_PO_greedy_betaLoss)
        with open(lamFile('DPP_PO_greedy'),'a') as f:
            f.write("%15.10f    " % optLam)
        print "DPP_PO_greedy MSE: %f    BETA_LOSS: %f   OPT_LAM: %f  TOTAL: %d" % (DPP_PO_greedy_mse, DPP_PO_greedy_betaLoss, optLam, sum(DPP_PO_greedy.gamma))


        ########################################
        ## OLSR:                              ##
        ##  - No Regularization               ##
        ##  - Complete Marginalization        ##
        ##  - Optimized hyperparameters       ##     
        ########################################

        OLSR_beta = linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        OLSR_yhat = OLSR_beta.T.dot(X_test.T).T
        OLSR_mse  = sum((y_test.T[0] - OLSR_yhat.T[0]) ** 2)
        OLSR_betaLoss = max(abs(OLSR_beta - oracleBeta))[0]
        with open(mseFile('OLSR'),'a') as f:
            f.write("%15.10f    " % OLSR_mse)
        with open(betaFile('OLSR'),'a') as f:
            f.write("%15.10f    " % OLSR_betaLoss)
        print "OLSR MSE: %f   BETA_LOSS: %f" % (OLSR_mse, OLSR_betaLoss)



        ########################################
        ## RIDGE:                             ##
        ##  - l2 Regularization               ##
        ##  - Complete Marginalization        ##
        ##  - Optimized hyperparameters       ##     
        ########################################

        ridgeLams = np.logspace(-5,6,200)

        def Learn(lam):
            ridge = Ridge(alpha=lam,fit_intercept=False,copy_X=True)
            ridge.fit(X_train,y_train)
            return ridge

        
        optLam = ExperimentUtils.gridSearch1D(ridgeLams, Learn, Eval, MAX=False)

        
        ridge = Ridge(alpha=optLam,fit_intercept=False,copy_X=True)
        ridge.fit(X,y)

        ridge_yhat = ridge.predict(X_test)
        ridge_mse = sum((y_test - ridge_yhat) ** 2)
        ridge_beta = ridge.coef_.T
        ridge_betaLoss = max(abs(ridge_beta - oracleBeta))[0]
        with open(mseFile('RIDGE'),'a') as f:
            f.write("%15.10f    " % ridge_mse)
        with open(betaFile('RIDGE'),'a') as f:
            f.write("%15.10f    " % ridge_betaLoss)
        with open(lamFile('RIDGE'),'a') as f:
            f.write("%15.10f    " % optLam)
        print "RIDGE MSE: %f    BETA_LOSS: %f   OPT_LAM: %f" % (ridge_mse, ridge_betaLoss, optLam)


        ########################################
        ## LASSO:                             ##
        ##  - l1 Regularization               ##
        ##  - Complete Marginalization        ##
        ##  - Optimized hyperparameters       ##     
        ########################################

        lassoLams = np.logspace(-5,6,200)

        def Learn(lam):
            lasso = Lasso(alpha=lam,fit_intercept=False,copy_X=True,max_iter=1.e7,tol=.0001)
            lasso.fit(X_train,y_train)
            return lasso

        
        optLam = ExperimentUtils.gridSearch1D(lassoLams, Learn, Eval, MAX=False,verbose=False)

        
        lasso = Lasso(alpha=optLam,fit_intercept=False,copy_X=True,max_iter=1.e7,tol=.0001)
        lasso.fit(X,y)

        lasso_yhat = np.array([lasso.predict(X_test)]).T
        lasso_mse = sum((y_test - lasso_yhat) ** 2)
        lasso_beta = np.array([lasso.coef_]).T
        lasso_betaLoss = max(abs(lasso_beta - oracleBeta))[0]
        with open(mseFile('LASSO'),'a') as f:
            f.write("%15.10f    " % lasso_mse)
        with open(betaFile('LASSO'),'a') as f:
            f.write("%15.10f    " % lasso_betaLoss)
        with open(lamFile('LASSO'),'a') as f:
            f.write("%15.10f    " % optLam)
        print "LASSO MSE: %f   BETA_LOSS: %f   OPT_LAM: %f" % (lasso_mse, lasso_betaLoss, optLam)


        ############
        ## Oracle ##
        ############

        gammaStar = dataGen.gamma
        XgamStar = DPPutils.columnGammaZero(X,gammaStar)
        Xreduc = DPPutils.gammaRM2D(XgamStar.T.dot(XgamStar),gammaStar)
        invOracle = DPPutils.addback_RC(linalg.inv(Xreduc),gammaStar)
        ORACLE_betaHAT = invOracle.dot(XgamStar.T).dot(y)
        ORACLE_yhat = ORACLE_betaHAT.T.dot(X_test.T).T
        ORACLE_mse  = sum((y_test.T[0] - ORACLE_yhat.T[0]) ** 2)
        ORACLE_betaLoss = max(abs(ORACLE_betaHAT - oracleBeta))[0]

        with open(mseFile('ORACLE'),'a') as f:
            f.write("%15.10f    " % ORACLE_mse)
        with open(betaFile('ORACLE'),'a') as f:
            f.write("%15.10f    " % ORACLE_betaLoss)
        print "ORACLE MSE: %f   BETA_LOSS: %f" % (ORACLE_mse, ORACLE_betaLoss)

    with open(mseFile('DPP_PO'),'a') as f:
        f.write("\n")
    with open(mseFile('DPP_PO_greedy'),'a') as f:
        f.write("\n")
    with open(mseFile('OLSR'),'a') as f:
        f.write("\n")
    with open(mseFile('RIDGE'),'a') as f:
        f.write("\n")
    with open(mseFile('LASSO'),'a') as f:
        f.write("\n")
    with open(mseFile('ORACLE'),'a') as f:
        f.write("\n")

    with open(betaFile('DPP_PO'),'a') as f:
        f.write("\n")
    with open(betaFile('DPP_PO_greedy'),'a') as f:
        f.write("\n")
    with open(betaFile('OLSR'),'a') as f:
        f.write("\n")
    with open(betaFile('RIDGE'),'a') as f:
        f.write("\n")
    with open(betaFile('LASSO'),'a') as f:
        f.write("\n")
    with open(betaFile('ORACLE'),'a') as f:
        f.write("\n")

    with open(lamFile('DPP_PO'),'a') as f:
        f.write("\n")
    with open(lamFile('DPP_PO_greedy'),'a') as f:
        f.write("\n")
    with open(lamFile('RIDGE'),'a') as f:
        f.write("\n")
    with open(lamFile('LASSO'),'a') as f:
        f.write("\n")     

