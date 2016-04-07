#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Creates exactly, the dataset used by Kojima and Komaki (2014).
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

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import math


#########################################################################

 
#########################################################################
###
### GENERATE_X
###
### Last Updated: 4/6/16
###

def genX(**kwargs):
    """
        Params: None
        Output: n datapoints based on Kojima and Komaki's Numerical Data
    """

    n = 400

    for k,v in kwargs.items():
        if k == 'n':
            n = v
    
    data_dist  = stats.multivariate_normal(np.zeros(n),np.eye(n))
    X = np.zeros((n,6))
    for i in range(6):
        X[:,i] = data_dist.rvs()
    X[:,3] = X[:,3] * 0.1 + X[:,0] + X[:,1]
    X[:,4] = X[:,4] * 0.1 + X[:,1] + X[:,2]
    X[:,5] = X[:,5] * 0.1 + X[:,2] + X[:,0]
    # X[:,7] = X[:,7] * 0.1 + X[:,1] + X[:,2]
    # X[:,8] = X[:,8] * 0.1 + X[:,1] + X[:,3]
    # X[:,9] = X[:,9] * 0.1 + X[:,2] + X[:,3]
    
    return X


#########################################################################


#########################################################################
###
### GENERATE_Y
###
### Last Updated: 4/6/16
###

def genY(X):
    """
        Params: X, created as in Kojima and Komaki's Numerical Data
        Output: Sampling of Y, per Kojima and Komaki's simulations 
    """
    n = X.shape[0]
    var = 0.81
    beta = np.array([[1.0],[-1.0],[0.],[0.],[0.],[0.]])
    noise_dist = stats.multivariate_normal(np.zeros(n),var*np.eye(n))
    eps = np.array([noise_dist.rvs()]).T
    y = X.dot(beta)+eps
    return y

#########################################################################


if __name__=='__main__':
    n = 400
    X = genX()
    y = genY(X)
    y2 = genY(X)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    for i in range(n):
        ax.scatter(X[i,0],X[i,1],y[i,0],c='blue')
        ax.scatter(X[i,0],X[i,1],y2[i,0],c='red')
    X,Y = np.meshgrid(np.linspace(-4,4,10),np.linspace(-4,4,10))
    Z = X - Y
    ax.plot_wireframe(X,Y,Z)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title('Kojima Komaki Numerical Data')
    plt.show()

