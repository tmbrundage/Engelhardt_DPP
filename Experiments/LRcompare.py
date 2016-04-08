#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Linear Regression Comparison Experiment
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
mainpath = "/u/tobrund/Engelhardt_DPP/"
sys.path.append(os.path.abspath(mainpath))
import time
import datetime

import numpy as np
import scipy.linalg as linalg

from Experiments.VI import VI

import Experiments.VI_MPM.BN_a0 as MPM_a0
import Experiments.VI_MPM.BN_b0 as MPM_b0
import Experiments.VI_MPM.BN_c  as MPM_c
import Experiments.VI_MPM.BN_theta_MPM as MPM_theta
import Experiments.VI_MPM.BN_gamma_MPM as MPM_gamma

import Experiments.VI_PPM.BN_var as PPM_var
import Experiments.VI_PPM.BN_theta_PPM as PPM_theta
import Experiments.VI_PPM.BN_gamma_PPM as PPM_gamma

import Experiments.VI_HP.BN_theta_HP as HP_theta
import Experiments.VI_HP.BN_gamma_HP as HP_gamma

import DataGeneration.KojimaKomakiDataGen as KKData
import Utils.DPPutils as DPPutils
import Utils.Memoizer as Memoizer
import Utils.ExperimentUtils as ExperimentUtils

from sklearn.linear_model import Ridge, Lasso

#########################################################################

ns = [10,20,50,100,200,500,1000,2000,5000]
n_test = 1000

##################
## Housekeeping ##
##################

loggingDirectory = 'Logs/%s/' % datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')

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
DPP1_log = "%sDPP1.txt" % outputDir
DPP2_log = "%sDPP2.txt" % outputDir
OLSR_log = "%sOLSR.txt" % outputDir
RIDGE_log = "%sRIDGE.txt" % outputDir
LASSO_log = "%sLASSO.txt" % outputDir
ORACLE_log = "%sORACLE.txt" % outputDir

oracleBeta = np.array([[1.,-1.,0.,0.,0.,0.]]).T

for n in ns:
    ####################
    ## Establish Data ##
    ####################

    print "<<<<< N = %d >>>>>" % n

    X = KKData.genX(n=n)
    p = X.shape[1]
    y = KKData.genY(X)

    X_test = KKData.genX(n=n_test)
    y_test = KKData.genY(X_test)

    val_size = int(0.1 * X.shape[0])
    X_val = X[0:val_size,:]
    y_val = y[0:val_size,:]
    X_train = X[val_size:,:]
    y_train = y[val_size:,:]

    
    ########################################
    ## DPP:                               ##
    ##  - Exponential Parameterization    ##
    ##  - Full Pre-Marginalization        ##
    ##  - Variational Hyperparameter MLE  ##     
    ########################################

    # Network Variables:
    DPP1_bnv = {}
    DPP1_bnv['a0'] = MPM_a0.BN_a0(val = 525.0)
    DPP1_bnv['b0'] = MPM_b0.BN_b0(val = 425.0)
    DPP1_bnv['c'] = MPM_c.BN_c()
    DPP1_bnv['theta'] = MPM_theta.BN_theta(param=p,prior='gaussian')
    DPP1_bnv['gamma'] = MPM_gamma.BN_gamma(param=p)

    lams = [1.0,5.,10.,50.,100.]#[0.0,1.0,2.0,5.0,10.,15.,25.,50.,75.,100.]

    def Learn(lam):
        DPP1_hp = {}
        DPP1_hp['lam_gamma'] = lam
        DPP1 = VI(X_train,y_train,DPP1_hp,DPP1_bnv,dir=loggingDirectory,logging=True,max_T=5e2,inner_T=3,verbose=False)
        DPP1.variationalInference()
        return DPP1

    # print "DPP1>>>>>>>>>>>>"
    optLam = ExperimentUtils.gridSearch1D(lams, Learn, Eval, MAX=False)

    # Hyperparameters:
    DPP1_hp = {}
    DPP1_hp['lam_gamma'] = optLam

    DPP1 = VI(X,y,DPP1_hp,DPP1_bnv,dir=loggingDirectory,logging=True,max_T=2.5e3,inner_T=3,verbose=False)
    DPP1.variationalInference()

    DPP1_yhat = DPP1.predict(X_test)
    DPP1_mse = sum((y_test.T[0] - DPP1_yhat.T[0]) ** 2)
    DPP1_betaLoss = max(abs(DPP1.getBeta() - oracleBeta))[0]
    with open(DPP1_log,'a') as f:
        f.write("%15.10f    %15.10f    %15.10f\n" % (DPP1_mse, DPP1_betaLoss, optLam))
    print "MLE_DPP MSE: %f   OPT_LAM: %f    BETA_LOSS: %f" % (DPP1_mse, optLam,DPP1_betaLoss)

  

    ########################################
    ## DPP:                               ##
    ##  - Exponential Parameterization    ##
    ##  - Partial Pre-Marginalization     ##
    ##  - Stationary Hyperparameters      ##     
    ########################################

    # Hyperparameters:
    DPP2_hp = {}
    DPP2_hp['lam_gamma'] = 10.0
    DPP2_hp['a0'] = 1.0
    DPP2_hp['b0'] = 2.0
    DPP2_hp['c']  = 2.0

    # Network Variables:
    DPP2_bnv = {}
    DPP2_bnv['theta'] = HP_theta.BN_theta(param=p,prior='gaussian')
    DPP2_bnv['gamma'] = HP_gamma.BN_gamma(param=p)


    # lams = [0.0,1.0,2.0,5.0,10.,15.,25.,50.,75.,100.]

    def Learn(lam):
        DPP2_hp['lam_gamma'] = lam
        DPP2 = VI(X_train,y_train,DPP2_hp,DPP2_bnv,dir=loggingDirectory,logging=True,max_T=5e2,inner_T=3,verbose=False)
        DPP2.variationalInference()
        return DPP2

    
    optLam = ExperimentUtils.gridSearch1D(lams, Learn, Eval, MAX=False)

    # Hyperparameters:
    DPP2_hp['lam_gamma'] = optLam

    DPP2 = VI(X,y,DPP2_hp,DPP2_bnv,dir=loggingDirectory,logging=True,max_T=2.5e3,inner_T=3,verbose=False)
    DPP2.variationalInference()

    DPP2_yhat = DPP2.predict(X_test)
    DPP2_mse = sum((y_test.T[0] - DPP2_yhat.T[0]) ** 2)
    DPP2_betaLoss = max(abs(DPP2.getBeta() - oracleBeta))[0]
    with open(DPP2_log,'a') as f:
        f.write("%15.10f    %15.10f    %15.10f\n" % (DPP2_mse, DPP2_betaLoss, optLam))
    print "HP_DPP MSE: %f   OPT_LAM: %f    BETA_LOSS: %f" % (DPP2_mse, optLam, DPP2_betaLoss)


    ########################################
    ## OLSR:                              ##
    ##  - No Regularization               ##
    ##  - Complete Marginalization        ##
    ##  - Optimized hyperparameters       ##     
    ########################################

    OLSR_beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    OLSR_yhat = OLSR_beta.T.dot(X_test.T).T
    OLSR_mse  = sum((y_test.T[0] - OLSR_yhat.T[0]) ** 2)
    OLSR_betaLoss = max(abs(OLSR_beta - oracleBeta))[0]
    with open(OLSR_log,'a') as f:
        f.write("%15.10f    %15.10f\n" % (OLSR_mse, OLSR_betaLoss))
    print "OLSR MSE: %f   BETA_LOSS: %f" % (OLSR_mse, OLSR_betaLoss)



    ########################################
    ## RIDGE:                             ##
    ##  - l2 Regularization               ##
    ##  - Complete Marginalization        ##
    ##  - Optimized hyperparameters       ##     
    ########################################

    ridgeLams = np.logspace(-5,6,23)

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
    with open(RIDGE_log,'a') as f:
        f.write("%15.10f    %15.10f    %15.10f\n" % (ridge_mse, ridge_betaLoss, optLam))
    print "RIDGE MSE: %f   OPT_LAM: %f    BETA_LOSS: %f" % (ridge_mse, optLam, ridge_betaLoss)


    ########################################
    ## LASSO:                             ##
    ##  - l2 Regularization               ##
    ##  - Complete Marginalization        ##
    ##  - Optimized hyperparameters       ##     
    ########################################

    lassoLams = np.logspace(-5,6,23)

    def Learn(lam):
        lasso = Lasso(alpha=lam,fit_intercept=False,copy_X=True,max_iter=5.e3)
        lasso.fit(X_train,y_train)
        return lasso

    
    optLam = ExperimentUtils.gridSearch1D(lassoLams, Learn, Eval, MAX=False,verbose=False)

    
    lasso = Lasso(alpha=optLam,fit_intercept=False,copy_X=True,max_iter=2.5e5)
    lasso.fit(X,y)

    lasso_yhat = np.array([lasso.predict(X_test)]).T
    lasso_mse = sum((y_test - lasso_yhat) ** 2)
    lasso_beta = np.array([lasso.coef_]).T
    lasso_betaLoss = max(abs(lasso_beta - oracleBeta))[0]
    with open(LASSO_log,'a') as f:
        f.write("%15.10f    %15.10f    %15.10f\n" % (lasso_mse, lasso_betaLoss, optLam))
    print "LASSO MSE: %f   OPT_LAM: %f   BETA_LOSS: %f" % (lasso_mse, optLam, lasso_betaLoss)


    ############
    ## Oracle ##
    ############

    gammaStar = np.array([[1,1,0,0,0,0]]).T
    XgamStar = DPPutils.columnGammaZero(X,gammaStar)
    Xreduc = DPPutils.gammaRM2D(XgamStar.T.dot(XgamStar),gammaStar)
    invOracle = DPPutils.addback_RC(np.linalg.inv(Xreduc),gammaStar)
    ORACLE_betaHAT = invOracle.dot(XgamStar.T).dot(y)
    ORACLE_yhat = ORACLE_betaHAT.T.dot(X_test.T).T
    ORACLE_mse  = sum((y_test.T[0] - ORACLE_yhat.T[0]) ** 2)
    ORACLE_betaLoss = max(abs(ORACLE_betaHAT - oracleBeta))[0]

    with open(ORACLE_log,'a') as f:
        f.write("%15.10f    %15.10f\n" % (ORACLE_mse, ORACLE_betaLoss))
    print "ORACLE MSE: %f   BETA_LOSS: %f" % (ORACLE_mse, ORACLE_betaLoss)
