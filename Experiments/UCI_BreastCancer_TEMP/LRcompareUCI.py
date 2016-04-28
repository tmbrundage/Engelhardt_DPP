#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: UCI Breast Cancer Data LR Comparison
####
####  Last updated: 4/20/16
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
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge, Lasso
from Experiments.UCI_BreastCancer import DataPrep
from Experiments.PO_PG import PO
from Experiments.PO_PG import PO_greedy
import Utils.ExperimentUtils as ExperimentUtils
#########################################################################




DP = DataPrep.DataPrep()

X_tr, X_te, y_tr, y_te = train_test_split(
DP.X_train,DP.y_train, test_size = .16)

guess = np.mean(DP.y_train)
baseline = sum((DP.y_test - guess) ** 2.)
print "BASELINE MSE: %f" % baseline

def Eval(learned):
    learned_yhat = learned.predict(X_te)
    learned_mse = sum((y_te - learned_yhat) ** 2)
    return learned_mse

loggingDirectory = 'Logs/' #%s/' % datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
outputDir = "%soutput/" % loggingDirectory
if not os.path.exists(outputDir):
    try:
        os.makedirs(outputDir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

fn_mse = 'mse.txt'
fn_sparsity = 'sparsity.txt'
with open("%s%s" % (outputDir,fn_mse),'a') as f:
    f.write("%15.10f    " % baseline)

###########
## LASSO ##
###########

lassoLams = np.logspace(-5,6,200)

def LearnLasso(lam):
    lasso = Lasso(alpha=lam,fit_intercept=False,copy_X=True,max_iter=1.e7,tol=0.0001)
    lasso.fit(X_tr,y_tr)
    return lasso

lassoOptLam = ExperimentUtils.gridSearch1D(lassoLams,LearnLasso,Eval,MAX=False,verbose=False)

lasso = Lasso(alpha=lassoOptLam,fit_intercept=False,copy_X=True,max_iter=1.e7,tol=.0001)
lasso.fit(DP.X_train,DP.y_train)

lasso_yhat = lasso.predict(DP.X_test)
lasso_mse = sum((DP.y_test - lasso_yhat)**2.)
lasso_sparsity = sum([1 for i in lasso.coef_ if i != 0.]) / float(X_tr.shape[1])
print "LASSO MSE: %f    TOTAL: %f" % (lasso_mse, lasso_sparsity)
with open("%s%s" % (outputDir,fn_mse),'a') as f:
    f.write("%15.10f    " % lasso_mse)
with open("%s%s" % (outputDir,fn_sparsity),'a') as f:
    f.write("%15.10f    " % lasso_sparsity)

###########
## RIDGE ##
###########

ridgeLams = np.logspace(-5,6,200)

def LearnRidge(lam):
    ridge = Ridge(alpha=lam,fit_intercept=False,copy_X=True)
    ridge.fit(X_tr,y_tr)
    return ridge

ridgeOptLam = ExperimentUtils.gridSearch1D(ridgeLams,LearnRidge,Eval,MAX=False)

ridge = Ridge(alpha=ridgeOptLam,fit_intercept=False,copy_X=True,max_iter=1.e7,tol=.0001)
ridge.fit(DP.X_train,DP.y_train)

ridge_yhat = ridge.predict(DP.X_test)
ridge_mse = sum((DP.y_test - ridge_yhat)**2.)
ridge_sparsity = sum([1 for i in ridge.coef_ if i != 0.]) / float(X_tr.shape[1])
print "RIDGE MSE: %f    TOTAL: %f" % (ridge_mse, ridge_sparsity)
with open("%s%s" % (outputDir,fn_mse),'a') as f:
    f.write("%15.10f    " % ridge_mse)
with open("%s%s" % (outputDir,fn_sparsity),'a') as f:
    f.write("%15.10f    " % ridge_sparsity)


#########
## DPP ##
#########

DPP_PO = PO.PO(DP.X_train,np.array([DP.y_train]).T,max_T = 500, GA_max_T = 200)
DPP_PO_yhat = DPP_PO.predict(DP.X_test)
DPP_PO_mse = sum((DP.y_test.T[0] - DPP_PO_yhat)**2.)
print "DPP_PO MSE: %f    TOTAL:%f    DROPPED:%d" % (DPP_PO_mse, sum(DPP_PO.gamma)/float(X_tr.shape[1]),len(DPP_PO.ignore))
with open("%s%s" % (outputDir,fn_mse),'a') as f:
    f.write("%15.10f    " % DPP_PO_mse)
with open("%s%s" % (outputDir,fn_sparsity),'a') as f:
    f.write("%15.10f    " % (sum(DPP_PO.gamma)/float(X_tr.shape[1])))

################
## GREEDY DPP ##
################

DPP_PO_greedy = PO_greedy.PO_greedy(DPP_PO,0.)
DPP_PO_greedy_yhat = DPP_PO_greedy.predict(DP.X_test)
DPP_PO_greedy_mse = sum((DP.y_test.T[0] - DPP_PO_greedy_yhat) **2.)
print "DPP_PO_greedy MSE: %f       TOTAL: %f" % (DPP_PO_mse, sum(DPP_PO_greedy.gamma)/float(X_tr.shape[1]))
with open("%s%s" % (outputDir,fn_mse),'a') as f:
    f.write("%15.10f\n" % DPP_PO_greedy_mse)
with open("%s%s" % (outputDir,fn_sparsity),'a') as f:
    f.write("%15.10f\n" % (sum(DPP_PO_greedy.gamma)/float(X_tr.shape[1])))
