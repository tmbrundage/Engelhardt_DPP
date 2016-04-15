#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Inspired by Kojima and Komaki (2014), creates a random 
####        dataset with collinearity. 
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
###
### IMPORTS
###

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import math
import itertools as itertools
from numpy.random import shuffle


#########################################################################



class CollinearDataGenerator(object):

    def __init__(self,**kwargs):
        self.p = 10
        self.collinear_p = .5
        self.shrink_to_noise = 0.1
        sparsity = .65

        for k,v in kwargs.items():
            if k == 'n':
                self.n = v
            elif k == 'p':
                self.p = int(v)
            elif k == 'collinear_p':
                self.collinear_p = v
            elif k == 'shrink_to_noise':
                self.shrink_to_noise = v
            elif k == 'sparsity':
                sparsity = v


        # Calculate how many features must be held as independent to create
        # The collinear features

        collinear_requested = int(round(self.collinear_p * self.p))
        independent_requested = self.p - collinear_requested
        self.collinear = collinear_requested
        self.independent = independent_requested
        while (self.collinear > sum(range(self.independent))):
            self.independent += 1
            self.collinear -= 1

        if self.collinear < collinear_requested:
            print "Warning: could not create requested proportion of pairwise covariant features."
            print "Requested (independent, collinear): (%d, %d). Using: (%d, %d)" % (independent_requested, collinear_requested, self.independent, self.collinear)

        # Standardize the reordering of features
        self.shuf = np.eye(self.p)
        shuffle(self.shuf)
        # print self.shuf

        nonzero_requested = self.p - int(round(sparsity * self.p))
        nonzero = min(self.independent, nonzero_requested)
        if nonzero < nonzero_requested:
            print "Warning: could not include %d nonzero coefficients with this level of colinearity." % nonzero_requested
            print "Using %d nonzero instead." % nonzero

        front = np.hstack((np.ones(nonzero), np.zeros(self.independent - nonzero)))
        shuffle(front)
        self.gamma = np.hstack((front, np.zeros(self.p - len(front)))).dot(self.shuf)

        # Get Beta*
        self.betaStar = self.genBeta()



    #########################################################################
    ###
    ### GET_X
    ###
    ### Last Updated: 4/10/16
    ###

    def getX(self,N):

        X = self.genX(N,self.p,self.independent,self.shrink_to_noise)
        Xshuf = X.dot(self.shuf)

        return Xshuf


    #########################################################################


    #########################################################################
    ###
    ### GET_Y
    ###
    ### Last Updated: 4/10/16
    ###

    def getY(self,X):

        n = X.shape[0]
        var = 0.5
        noise_dist = stats.multivariate_normal(np.zeros(n),var*np.eye(n))
        eps = np.array([noise_dist.rvs()]).T
        # print "EPS:"
        # print eps

        y = X.dot(self.betaStar) + eps

        return y


    #########################################################################


    #########################################################################
    ###
    ### GENERATE_X
    ###
    ### Last Updated: 4/10/16
    ###
    @staticmethod
    def genX(n,p,independent,shrink_to_noise):
        """
            Params: Number of datapoints, N, number of features, p, how many 
                    are just gaussian draws, independent.
            Output: n datapoints with a random number of features, a random
                    portion of which are collinear.
        """

        # Data is normally distributed with unit variance 
        # FOR NOW, FEATURES ARE ROWS
        data_dist = stats.multivariate_normal(np.zeros(n),np.eye(n))
        X = np.array([data_dist.rvs() for i in range(p)])
        # print "X_ORIGINAL"
        # print X

        # Make the present (independent) values in the (future) collinear feature
        # rows a noise factor
        X[independent:,:] = map(lambda x: x * shrink_to_noise, X[independent:,:])

        # Add the combinations -- we may not necessarily need all of them.
        comb = itertools.combinations(range(independent), 2)
        for i,(j,k) in enumerate(comb):
            if i >= p - independent:
                break
            X[independent + i,:] += X[j,:] + X[k,:]

        # Return features in columns
        return X.T



    #########################################################################



    #########################################################################
    ###
    ### GENERATE_SPARSE_BETA
    ###
    ### Last Updated: 4/10/16
    ###

    def genBeta(self,**kwargs):
        mu = np.zeros(self.p)
        cov = 5 * np.eye(self.p)

        for k,v in kwargs.items():
            if k == 'cov':
                cov = v * np.eye(self.p)
            elif k == 'mu':
                mu = v * np.ones(self.p)

        data_dist = stats.multivariate_normal(mu,cov)
        beta = data_dist.rvs()

        
        sparse_beta = beta * self.gamma
        return np.array([sparse_beta]).T

    #########################################################################


#########################################################################

if __name__ == '__main__':
    dataGen = CollinearDataGenerator(p = 6, collinear_p = .5)
    print "BETA:"
    betaStar = dataGen.betaStar
    print betaStar

    X = dataGen.getX(17)
    print "FINAL_X"
    print X

    y = dataGen.getY(X)
    print "FINAL_Y"
    print y


    



