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
mainpath = "/u/tobrund/Engelhardt_DPP"
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
import DataGeneration.KojimaKomakiDataGen as KKData

import numpy as np
import scipy.linalg as linalg

#########################################################################

####################
## Establish Data ##
####################
n = 400
X = KKData.genX(n=n)
p = X.shape[1]
y = KKData.genY(X)
# Set minmum value for b0
P = X.dot(np.linalg.inv(X.T.dot(X)+np.eye(p))).dot(X.T)
lam_max = linalg.eigvalsh(P,eigvals=(n - 1, n - 1),type=2, overwrite_a=True)[0]
b0min = 0.5 * (lam_max - 1.0) * y.T.dot(y)[0][0]
assert(b0min <= 0.0)

#########################
## Set Hyperparameters ##
#########################
hp = {}
hp['lam_gamma'] = 10.0


##############################
## Create Network Variables ##
##############################

bnv = {}
bnv['a0'] = BN_a0(val = 525.0)
bnv['b0'] = BN_b0(param=b0min, val = 425.0)
bnv['c']  = BN_c()
bnv['theta'] = BN_theta(param=p,prior='gaussian')
bnv['gamma'] = BN_gamma(param=p)



loggingDirectory = 'Logs/%s/' % datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
experiment = VI(X,y,hp,bnv,dir=loggingDirectory,logging=True,max_T=1e4,inner_T=3)
experiment.variationalInference()

c = bnv['c'].val_getter()
gamma = bnv['gamma'].val_getter()
Xgam = DPPutils.columnGammaZero(X,gamma)
inv = np.linalg.inv(c * np.eye(p)+Xgam.T.dot(Xgam))
beta = y.T.dot(Xgam).dot(inv).T

gammaStar = np.array([[1,1,0,0,0,0]]).T
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


