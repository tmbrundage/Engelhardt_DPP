#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: UCI Breast Cancer Data Prepper
####
####  Last updated: 4/30/16
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
from sklearn.cross_validation import KFold

#########################################################################




fn = 'wpbc.data'

df = pd.read_csv(fn,header=None)
# print df.loc[:,1:2]

# df.loc[df[1]=='N',2] *= -1.
del df[34]

k = 10
kf = KFold(n=df.shape[0],n_folds=k,shuffle=True)
#   (X,y) |--> df.loc[:,3:], df.loc[:,2]
for i, (tr_idx, te_idx) in enumerate(kf):
    sys.stdout.write("Creating Set %d     \r" % i)
    sys.stdout.flush()
    setDir = 'Fold%d/' % i
    os.makedirs(setDir)

    X_train = df.loc[tr_idx,3:]
    y_train = df.loc[tr_idx,2]
    X_test  = df.loc[te_idx,3:]
    y_test  = df.loc[te_idx,2]

    X_mu = X_train.mean()
    X_range = X_train.max() - X_train.min()

    X_train -= X_mu
    X_train /= X_range
    X_test -= X_mu
    X_test /= X_range
    y_mu = y_train.mean()
    y_range = y_train.max() - y_train.min()        
    y_train -= y_mu
    y_train /= y_range
    y_test -= y_mu
    y_test /= y_range

    np.save('%sX_tr' % setDir,X_train,allow_pickle=True)
    np.save('%sy_tr' % setDir,y_train,allow_pickle=True)
    np.save('%sX_te' % setDir,X_test,allow_pickle=True)
    np.save('%sy_te' % setDir,y_test,allow_pickle=True)
    with open('%sNormValues.txt' % setDir,'a') as f:
        f.write('y_range = %15.10f\n' % y_range)
        f.write('y_mu    = %15.10f\n' % y_mu)
        f.write('X_range = [')
        for x in X_range:
            f.write('%15.10f  ' % x)
        f.write(']\n')
        f.write('X_mu    = [')
        for x in X_mu:
            f.write('%15.10f  ' % x)
        f.write(']\n')
        


