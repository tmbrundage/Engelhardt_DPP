#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Theta Prior Likelihoods and Gradients
####
####  Last updated: 3/17/16
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
mainpath = "/Users/Ted/__Engelhardt/Code"
sys.path.append(os.path.abspath(mainpath))
import numpy as np
import scipy.stats as stats
from Experiments.VI import VI

#########################################################################


#########################################################################
###
### UNIFORM_LIKELIHOOD_AND_GRADIENT
###
### Last Updated: 3/17/16
###

def uniformLikelihood(state):
    if not issubclass(type(state), VI):
        raise StateError('State must be given in terms of VI object, not as %s.' % type(state).__name__)
    return 0.0

def gradUniformLikelihood(state):
    if not issubclass(type(state), VI):
        raise StateError('State must be given in terms of VI object, not as %s.' % type(state).__name__)
    gradLog = np.zeros((state.p,1))
    return gradLog

#########################################################################


#########################################################################
###
### LAPLACE_LIKELIHOOD_AND_GRADIENT
###
### Last Updated: 4/5/16
###

def laplaceLogLikelihood(state):
    if not issubclass(type(state), VI):
        raise StateError('State must be given in terms of VI object, not as %s.' % type(state).__name__)
    theta = state.bnv['theta'].val_getter()
    L = -1.0 * sum(abs(theta.T))[0]
    return L

def gradLaplaceLogLikelihood(state):
    if not issubclass(type(state), VI):
        raise StateError('State must be given in terms of VI object, not as %s.' % type(state).__name__)
    theta = state.bnv['theta'].val_getter()
    grdL = -1.0 * np.sign(theta)
    return grdL

#########################################################################



#########################################################################
###
### GAUSSIAN_LIKELIHOOD_AND_GRADIENT
###
### Last Updated: 4/5/16
###

def gaussianLogLikelihood(state):
    if not issubclass(type(state), VI):
        raise StateError('State must be given in terms of VI object, not as %s.' % type(state).__name__)
    theta = state.bnv['theta'].val_getter()
    # assume mean = np.zeros(state.p)
    # assume cov  = np.eye(state.p)
    # dist = stats.multivariate_normal(mean, cov)
    # L = dist.logpdf(state.bnv['theta'].val_getter().T[0])
    # For this mean and covariance, taking the log, we can drop the constant,
    # and just have the following:
    L = -0.5 * theta.T.dot(theta)[0][0]
    return L

def gradGaussianLogLikelihood(state):
    if not issubclass(type(state), VI):
        raise StateError('State must be given in terms of VI object, not as %s.' % type(state).__name__)
    theta = state.bnv['theta'].val_getter()
    # assume mean = np.zeros(state.p)
    # assume cov  = np.eye(state.p)
    # dist = stats.multivariate_normal(mean, cov)
    # For this mean and covariance, taking the log, we can drop the constant,
    # and just have the following:
    gradL = -0.5 * theta
    return gradL

#########################################################################


#########################################################################
###
### EXPECTATION_L0_LIKELIHOOD_AND_GRADIENT
###
### Last Updated: 4/5/16
###
### Note: The expectation of the cardinality of gamma is given by the 
###       trace of K. Since this is a prior, we do not want influence of
###       X here, so we will not be using the full K kernel, but rather
###       use the simple kernel, assuming perfectly independent features,
###       or X.T.dot(X) = np.eye(state.p)
###       Thus, our penalty will be given based on the expectation of 
###       the cardinality implied by theta.
###

def expectationL0LogLikelihood(state):
    if not issubclass(type(state), VI):
        raise StateError('State must be given in terms of VI object, not as %s.' % type(state).__name__)
    theta = state.bnv['theta'].val_getter()
    # Given X.T.dot(X) = np.eye(p), L = diag(theta_i^2)
    # Eigenvalues are clearly theta_i^2 -- scale by theta_i^2 + 1
    trace = sum(map(lambda x: x**2 / (x**2 + 1), theta))[0]
    return np.log(trace)

def gradExpectationL0LogLikelihood(state):
    if not issubclass(type(state), VI):
        raise StateError('State must be given in terms of VI object, not as %s.' % type(state).__name__)
    theta = state.bnv['theta'].val_getter()
    scale = np.exp(-1.0 * expectationL0LogLikelihood(state))
    grdL = map(lambda x: 2.0 * x / (x ** 2 + 1.0) ** 2, theta)
    return grdL * scale

#########################################################################




