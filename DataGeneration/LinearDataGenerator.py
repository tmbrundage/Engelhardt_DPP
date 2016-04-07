#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Linear Data Generator, inspired by experiments in section 4
####        of Kojima and Komaki (2014).
####
####  Last updated: 3/19/16
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
import sys
import math


#########################################################################



class LinearDataGenerator(object):

    #########################################################################
    ###
    ### GENERATE_BETA
    ###
    ### Last Updated: 3/3/16
    ###

    @staticmethod
    def generate_beta(p,var):
        """
            Params: p gives the number of features, or dimension of beta
                    c gives the scaling of the variance for our COV matrix
            Output: Returns beta, drawn randomly from a zero mean, p-dimensional
                    gaussian with covariance defiend as (c/v)I_p, as defined in
                    Bornn, 2014 and the Linear Regression section of my notes. 
        """
        cov = (var) * np.eye(p)
        mu = np.zeros(p)
        beta = np.array([np.random.multivariate_normal(mu, cov)]).T
        return beta

    #########################################################################


    #########################################################################
    ###
    ### GENERATE_GAMMA
    ###
    ### Last Updated: 3/3/16
    ###

    @staticmethod
    def generate_gamma(p,gam_k):
        """
            Params: p gives the number of features, or the dimension of generate_gamma
                    gam_k gives the desired l1 norm of gam, or the number of selected
                    features.
            Output: Returns a random permutation of gam_k ones and (p-gam_k) ones 
                    as an inclusion vector
        """
        if gam_k > p:
            raise NameError('Desired selected featurese cannot exceed number of features.')
        ordered = np.hstack((np.ones(gam_k),np.zeros(p-gam_k)))
        gamma = np.array([np.random.permutation(ordered)]).T     
        return gamma

    #########################################################################
    


    #########################################################################
    ###
    ### INITIALIZER
    ###
    ### Last Updated: 3/3/16
    ###

    def __init__(self,**kwargs):
        self.p = 6
        self.n = 1000

        self.gam_k = 2 # Desired cardinality of gamma
        self.a0 = 1.0
        self.b0 = 1.0
        
        self.width = 1.0

        for name,value in kwargs.items():
            if name == 'p':
                self.p = int(value)
            if name == 'n':
                self.n = int(value)
            if name == 'gam_p':
                self.gam_p = float(value)
            if name == 'a0':
                self.a0 = float(value)
            if name == 'b0':
                self.b0 = float(value)
            if name == 'width':
                self.width = float(value)

        self.var = self.b0 / (self.a0 + 1) # MLE (mode) of inv-gamma(a,b) is b/(a+1)

        """
            WRITE CHECKS FOR ALL OF THESE DATATYPES
        """

        # Generate coefficients
        self.beta = LinearDataGenerator.generate_beta(self.p,self.var)
        self.gam  = LinearDataGenerator.generate_gamma(self.p,self.gam_k)
        self.bg   = np.multiply(self.beta,self.gam)

        # Generate data and labels
        self.X, self.y = self.getTestData(self.n)
        
        
    #########################################################################



    #########################################################################
    ###
    ### GET_UNIFORM_TEST_DATA
    ###
    ### Last Updated: 3/3/16
    ###

    def getUniformTestData(self,n):
        """
            Params: Given a LinearDataGenerator object, this creates another
            Output: n datapoints from the same distribution. 
        """
        data_dist  = stats.multivariate_normal(np.zeros(self.p),self.width*np.eye(self.p))
        noise_dist = stats.multivariate_normal(np.zeros(self.n),self.var*np.eye(self.n))
        X = np.array([data_dist.rvs() for i in range(self.n)])
        y = np.array([[X[i,:].dot(self.bg)[0] for i in range(self.n)]]).T \
             + np.array([noise_dist.rvs()]).T
        return X, y

    #########################################################################


#########################################################################
###
### MAIN
###
### Last Updated: 3/3/16
###
### Note: For tests
###


# if __name__ == '__main__':
#     ldg = LinearDataGenerator()
#     print ldg.y
