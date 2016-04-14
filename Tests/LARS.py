import os
import sys
mainpath = "/Users/Ted/__Engelhardt/Engelhardt_DPP/"
sys.path.append(os.path.abspath(mainpath))
import time
import datetime

import numpy as np
import scipy.linalg as linalg

import DataGeneration.CollinearDataGenerator as CDG

from sklearn.linear_model import Lars, lars_path

ns = [10,20,50,100,200,500,1000]

for n in ns:
    print "<<<<< N = %d >>>>>" % n

    cdg = CDG.CollinearDataGenerator(p = 20,sparsity=.8)
    X = cdg.getX(n)
    p = X.shape[1]
    y = cdg.getY(X)

    print cdg.gamma

    val_size = int(0.1 * X.shape[0])
    X_val = X[0:val_size,:]
    y_val = y[0:val_size,:]
    X_train = X[val_size:,:]
    y_train = y[val_size:,:]

    lars = Lars(n_nonzero_coefs=2)
    lars.fit(X,y)
    # print lars.coef_

    alphas, order, coefs = lars_path(X,y.T[0],verbose=True)
    # print alphas
    print order
    magnitudes = sorted(list(enumerate(coefs[:,-1])),key=lambda x: x[1])
    magnitudes = map(lambda x: x[0],magnitudes)
    print magnitudes
    # print coefs
    quantities = coefs[:,-1]
    quantities = np.array([quantities[i] for i in order])
    # print quantities
    total = sum(abs(quantities))
    # # print total
    cumsum = np.array(reduce(lambda a, x: a + [a[-1] + abs(x)], quantities[1:],[abs(quantities[0])]))
    print cumsum / total
