#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Variational Inference Maximal Pre-Marginalization Experiment
####        With smart-init.
####
####  Last updated: 4/10/16
####
####  Notes and disclaimers:
####    - Use only numpy.ndarray, not numpy.matrix to avoid any confusion
####    - If something is a column vector - MAKE IT A COLUMN VECTOR. Makes 
####          manipulation annoying, but it keeps matrix algebra logical. 
####
#########################################################################


#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Variational Inference Maximal Pre-Marginalization Experiment
####
####  Last updated: 4/6/16
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
mainpath = "/Users/Ted/__Engelhardt/Engelhardt_DPP"
sys.path.append(os.path.abspath(mainpath))
import time
import datetime

from Experiments.VI import VI
from Experiments.VI_MPM.BN_a0 import BN_a0
from Experiments.VI_MPM.BN_b0 import BN_b0
from Experiments.VI_MPM.BN_c  import BN_c
from Experiments.VI_MPM.BN_theta_MPM import BN_theta
from Experiments.VI_MPM.BN_gamma_MPM import BN_gamma
import Utils.DPPutils as DPPutils
import Utils.Memoizer as Memoizer
import Utils.ExperimentUtils as ExperimentUtils
import DataGeneration.CollinearDataGenerator as CDG

import numpy as np
import scipy.linalg as linalg
from scipy.stats.stats import pearsonr


#########################################################################

####################
## Establish Data ##
####################
n = 400
cdg = CDG.CollinearDataGenerator(p = 20,sparsity=.8)
X = cdg.getX(n)
p = X.shape[1]
y = cdg.getY(X)
with open('TEMPX.txt','w') as f:
    f.write('%s'% repr(X))

print cdg.gamma
# for i in range(p):
#     print pearsonr(y.T[0],X[:,i])



val_size = int(0.1 * X.shape[0])
X_val = X[0:val_size,:]
y_val = y[0:val_size,:]
X_train = X[val_size:,:]
y_train = y[val_size:,:]


# Set minmum value for b0
P = X.dot(np.linalg.inv(X.T.dot(X)+np.eye(p))).dot(X.T)
lam_max = linalg.eigvalsh(P,eigvals=(n - 1, n - 1),type=2, overwrite_a=True)[0]
b0min = 0.5 * (lam_max - 1.0) * y.T.dot(y)[0][0]
assert(b0min <= 0.0)

#####################
## Estimate Values ##
#####################
OLSR_beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
OLSR_yhat = OLSR_beta.T.dot(X.T).T
eps_var = np.var((y-OLSR_yhat).T[0])
beta_var = np.var(OLSR_beta.T[0])
est_c = eps_var / beta_var

est_aN = 1.0 + n * 0.5
est_bN = 1.0 + 0.5 * y.T.dot(np.eye(n) - X.dot(np.linalg.inv(X.T.dot(X) + est_c * np.eye(p))).dot(X.T)).dot(y)[0][0]

# print eps_var
# print aN
# print bN
# print bN / (aN + 1.0)




##############################
## Create Network Variables ##
##############################



bnv = {}
bnv['a0'] = BN_a0(val = est_aN,alpha=1.)
bnv['b0'] = BN_b0(param=b0min, val = est_bN,alpha=1.)
bnv['c']  = BN_c(val = est_c)
bnv['theta'] = BN_theta(param=p,prior='uniform',alpha=1.)
bnv['gamma'] = BN_gamma(param=p)

loggingDirectory = 'Logs/%s' % datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')


# lams = np.linspace(1.,20.,10)#[0.0,1.0,2.0,5.0,10.,15.,25.,50.,75.,100.]

# def Learn(lam):
#     DPP1_hp = {}
#     DPP1_hp['lam_gamma'] = lam
#     DPP1 = VI(X_train,y_train,DPP1_hp,bnv,dir=loggingDirectory,logging=True,max_T=2.5e2,inner_T=3,verbose=False)
#     DPP1.variationalInference()
#     return DPP1

# def Eval(learned):
#     learned_yhat = learned.predict(X_val)
#     learned_mse = sum((y_val - learned_yhat) ** 2)
#     return learned_mse
# optLam = ExperimentUtils.gridSearch1D(lams, Learn, Eval, MAX=False)


#########################
## Set Hyperparameters ##
#########################
hp = {}
hp['lam_gamma'] = 10


experiment = VI(X,y,hp,bnv,dir=loggingDirectory,logging=True,max_T=1e1,inner_T=3)
experiment.variationalInference()

c = bnv['c'].val_getter()
gamma = bnv['gamma'].val_getter()
Xgam = DPPutils.columnGammaZero(X,gamma)
inv = np.linalg.inv(c * np.eye(p)+Xgam.T.dot(Xgam))
beta = y.T.dot(Xgam).dot(inv).T

gammaStar = np.array([cdg.gamma]).T
XgamStar = DPPutils.columnGammaZero(X,gammaStar)
Xreduc = DPPutils.gammaRM2D(XgamStar.T.dot(XgamStar),gammaStar)
invOracle = DPPutils.addback_RC(np.linalg.inv(Xreduc),gammaStar)

compareFile = '%s/compareResults.txt' % loggingDirectory
with open(compareFile,'w') as f:
    f.write('MINE:\n')
    f.write('%s\n\n' % repr(beta))

    f.write('LSR:\n')
    f.write('%s\n\n' % np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y))

    f.write('Ridge:\n')
    f.write('%s\n\n' % np.linalg.inv(X.T.dot(X) + c * np.eye(p)).dot(X.T).dot(y))

    f.write('Oracle:\n')
    f.write('%s\n\n' % invOracle.dot(XgamStar.T).dot(y))
