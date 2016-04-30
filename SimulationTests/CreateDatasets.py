#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Script to Generate All Datasets
####
####  Last updated: 4/25/16
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
from copy import deepcopy as dc
import time
import datetime

mainpath = "/Users/Ted/__Engelhardt/Engelhardt_DPP"
sys.path.append(os.path.abspath(mainpath))

import numpy as np
import DataGeneration.CollinearDataGenerator as CDG

#########################################################################


N = 100

p = 40
s = 0.90
ns = [25,50,75,100,150,200,400]
n_test = 1000

for i in xrange(N):
    sys.stdout.write("Creating Set %d   \r" % (i))
    sys.stdout.flush()
    setDir = 'Set%02d/' % i
    os.makedirs(setDir)

    dataGen = CDG.CollinearDataGenerator(p=p,sparsity=s)
    betaStar = dataGen.betaStar
    gammaStar = dataGen.gamma
    np.save('%sbetaStar'%setDir,betaStar,allow_pickle=True)
    np.save('%sgammaStar'%setDir,gammaStar,allow_pickle=True)

    for n in ns:
        nDir = '%sn_%03d/'% (setDir,n)
        os.makedirs(nDir)
        X_tr = dataGen.getX(n)
        y_tr = dataGen.getY(X_tr)
        X_te = dataGen.getX(n_test)
        y_te = dataGen.getY(X_te)
        np.save('%sX_tr' % nDir,X_tr,allow_pickle=True)
        np.save('%sy_tr' % nDir,y_tr,allow_pickle=True)
        np.save('%sX_te' % nDir,X_te,allow_pickle=True)
        np.save('%sy_te' % nDir,y_te,allow_pickle=True)

