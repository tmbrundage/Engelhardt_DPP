#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: UCI Breast Cancer Data Prepper
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
mainpath = "/u/tobrund/Engelhardt_DPP"
sys.path.append(os.path.abspath(mainpath))
import numpy as np
import pandas as pd 
from sklearn.cross_validation import train_test_split

#########################################################################


class DataPrep(object):


    #########################################################################
    ###
    ### INITIALIZER
    ###
    ### Last updated: 4/20/16
    ###

    def __init__(self):
        fn = 'wpbc.data'
        # names = ['ID',
        #          'Outcome',
        #          'Time',
        #          'radius',
        #          'texture',
        #          'perimeter',
        #          'area',
        #          'smoothness',
        #          'compactness',
        #          'concavity',
        #          'concave_points',
        #          'symmetry',
        #          'fractal_dimension',
        #          'size',
                 # 'lymph_status']
        df = pd.read_csv(fn,header=None)
        df.loc[df[1]=='N',2] *= -1.
        del df[34]
        # X = df.loc[:,3:]
        # print X.shape
        # print df.loc[:,3:]
        X_train, X_test, y_train, y_test = train_test_split(
            df.loc[:,3:], df.loc[:,2], test_size = 0.1)
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

        self.X_train = X_train.as_matrix()
        self.y_train = y_train.as_matrix()
        self.X_test = X_test.as_matrix()
        self.y_test = y_test.as_matrix()
        # print self.X_train.shape
        # print self.y_train.shape

        self.X_mu = X_mu
        self.X_range = X_range
        self.y_my = y_mu
        self.y_range = y_range


    #########################################################################



if __name__ == "__main__":
    dp = DataPrep()
