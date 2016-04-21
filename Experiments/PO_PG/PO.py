#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Parameter Optimizer
####
####  Last updated: 4/16/16
####
####  Notes and disclaimers:
####    - I implement Theta as an instance variable, rather than a BNV
####          given the optimizations that are possible when L and gradL
####          are computed simulatneously. 
####    - Gamma is selected via sampling
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
import numpy.matlib as matlib
import scipy.special as funcs
import scipy.linalg as linalg
import math as math
import random as random
import itertools as itertools
import DataGeneration.CollinearDataGenerator as CDG
import Utils.ExperimentUtils as ExperimentUtils
import Utils.DPPutils as DPPutils
import Utils.Memoizer as Memoizer
from Utils.BNV import BNV

from sklearn.linear_model import lars_path


#########################################################################





class PO(object):


    #########################################################################
    ###
    ### INITIALIZER
    ###
    ### Last Updated: 4/13/16
    ###


    def __init__(self,X,y,**kwargs):
        self.X = X
        self.y = y
        self.S = np.transpose(X).dot(X)
        (self.n, self.p) = self.X.shape
        if self.n <= 2.:
            raise BNV.ArgumentError("Not enough data: n = %d" % self.n)
        
        self.max_T = int(5e2)   # Number of draws of gamma
        self.minGamma = 10      # Minimum number of features to marginalize over
        self.alpha = 1.      # Learning rate on Theta Optimization
        self.tau = 1.e-3        # Tolerance on Theta Optimization
        self.GA_max_T = int(5e2)# Number of iterations on Theta Optimization
        self.n_converge = 10    # Number of updates < tau to determine convergence

        self.check = True
        self.verbose = True
        self.logging = True
        self.dir = 'Logs/%s/' % datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')

        for name,value in kwargs.items():
            if name == 'max_T':
                self.max_T = int(value)
            elif name == 'minGamma':
                self.minGamma = int(value)
            elif name == 'dir':
                self.dir = value
            elif name == 'logging':
                self.logging = value
            elif name == 'check':
                self.check = value
            elif name == 'alpha':
                self.alpha = float(value)
            elif name == 'tau':
                self.tau = float(tau)
            elif name == 'GA_max_T':
                self.GA_max_T = int(value)
            elif name == 'n_converge':
                self.n_converge = int(value)

        if self.check:
            assert(type(X) == np.ndarray)
            assert(type(y) == np.ndarray)
            assert(y.shape == (X.shape[0],1))


        if self.logging:
            if not os.path.exists(self.dir):
                try:
                    os.makedirs(self.dir)
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            self.thetaFN = "%stheta.txt" % self.dir

        self.memoizer = Memoizer.Memoizer(self.X, self.y, check=self.check)


        self.c = self.cRROpt()

        # print "c = %f" % self.c

        diffProj = self.memoizer.FDifferenceProjection(np.ones((self.p,1)),self.c)
        self.var = 2. * diffProj / (self.n - 2.)
        self.a0 = 1.
        self.b0 = self.var / (1. + self.a0)

        # print "var = %f" % self.var


        # self.larsTopGamma = self.larsSet(p=0.75,cap=10)
        self.larsTopGamma = self.larsSet()
        self.larsCap = 10
        # print self.larsTopGamma


        self.ignore = set()

        self.theta = np.zeros((self.p,1))
        self.gradientAscentTheta()

        # print self.theta

        self.gamma = self.gammaSamplingSelector()


        # Set Coefficients
        Xgam = DPPutils.columnGammaZero(self.X,self.gamma)
        Minv = linalg.inv(self.c * np.eye(self.p) + Xgam.T.dot(Xgam))
        self.beta = self.y.T.dot(Xgam).dot(Minv).T


    #########################################################################


    #########################################################################
    ###
    ### GAMMA_LIKELIHOOD
    ###
    ### Last Updated: 4/15/16
    ###
    ###

    def gammaLikelihood(self, gamma):
        # No need to include normalization of p(gamma|theta)
        pYGam = self.ALTpYconditionalGammaX(gamma)
        if pYGam == 0.:
            return sys.float_info.min
        pGamTheta = self.memoizer.FdetL(gamma,self.theta)
        return np.log(pYGam * pGamTheta)


    #########################################################################



    #########################################################################
    ###
    ### GAMMA_SAMPLING_SELECTOR
    ###
    ### Last Updated: 4/14/16
    ###
    ### Note: Returns the optimal value for theta marginalized over gammas 
    ###       generated from the LARS set.
    ###

    def gammaSamplingSelector(self):
        
        # Build the L ensembel with self's theta. Find the eigendecomposition
        expTheta = np.exp(0.5 * self.theta)
        coeffs = expTheta.dot(expTheta.T)
        L = coeffs * self.S

        eigVals, eigVecs = np.linalg.eigh(L)

        maxGamma = np.zeros((self.p,1))
        maxLikelihood = self.gammaLikelihood(maxGamma)

        tested = set()
        tested.add(repr(maxGamma))

        for sample in range(self.max_T):
            gammaTest = self.DPPSampler(eigVals, eigVecs)
            if repr(gammaTest) not in tested:
                likelihood = self.gammaLikelihood(gammaTest)
                if likelihood > maxLikelihood:
                    maxLikelihood = likelihood
                    maxGamma = gammaTest
                tested.add(repr(gammaTest))

        return maxGamma


    #########################################################################



    #########################################################################
    ###
    ### THETA_OPTIMIZER
    ###
    ### Last Updated: 4/14/16
    ###
    ### Note: Returns the optimal value for theta marginalized over gammas 
    ###       generated from the LARS set.
    ###

    def gradientAscentTheta(self):

        converging = 0

        for step in range(self.GA_max_T):
            # print "STEP %d" % step
            if converging >= self.n_converge:
                break

            if self.verbose and step % 100 == 0:
                # print "Theta Optimization, step: %d " % step
                # print "   %s" % repr(self.theta)
                if self.logging:
                    with open(self.thetaFN,'a') as f:
                        f.write('\n>>>>>>>>>>>>>>\nSTEP%d \n' % step)\

            L, gradL = self.theta_L_gradL_gamma(self.theta)
            self.theta += self.alpha * gradL

            if self.logging:
                with open(self.thetaFN,'a') as f:
                    f.write(' + %s\n%s\n' % (repr(gradL),repr(self.theta)))

            # If we're under the threshold, log it. Otherwise,
            # restart count.
            if max(abs(gradL)) < self.tau:
                converging += 1
            else: 
                converging = 0



    #########################################################################


    #########################################################################
    ###
    ### C_RR_OPT
    ###
    ### Last Updated: 4/13/16
    ###
    ### Note: Returns the optimal value for the regularization parameter in
    ###       ridge regression. Note - we're not splitting into a train and
    ###       validation set yet.
    ###

    def cRROpt(self, n=20,lam_min=-2,lam_max=3):
        # DOES IT MAKE A DIFFERENCE IF I DO THIS WITH ALL TRAINING DATA
        # OR DO I NEED TO FORCE A VALIDATION SET?
        # YES??
        val_size = int(0.1 * self.X.shape[0])
        X_val = self.X[0:val_size,:]
        y_val = self.y[0:val_size,:]
        X_train = self.X[val_size:,:]
        y_train = self.y[val_size:,:]

        def Eval(learned):
            learned_yhat = X_val.dot(learned)
            learned_mse = sum((y_val - learned_yhat) ** 2)
            return learned_mse

        def Learn(lam):
            inverse = linalg.inv(X_train.T.dot(X_train) + lam * np.eye(self.p))
            learned_beta = inverse.dot(X_train.T).dot(y_train)
            return learned_beta

        lams = np.logspace(lam_min,lam_max,n)

        opt_c = ExperimentUtils.gridSearch1D(lams,Learn,Eval)

        return opt_c


    #########################################################################



    #########################################################################
    ###
    ### LARS_SET
    ###
    ### Last Updated: 4/20/16
    ###
    ### Note: Returns the first features that capture proportion p of the 
    ###       "correlation" in LARS. These are not necessarily the variables
    ###       with the largest coefficients. They are taken in the order that
    ###       they were selected by LARS.
    ###

    def larsSet(self):#,p=.75,cap=10):
        alphas, order, coefs = lars_path(self.X,self.y.T[0])
        return order
        # magnitudes = np.array([abs(coefs[i,-1]) for i in order])
        # total = sum(magnitudes)

        # partialSums = np.array(reduce(lambda a, x: a + [a[-1] + x], magnitudes[1:], [magnitudes[0]]))

        # maxIdx = 1
        # while maxIdx < len(partialSums) and partialSums[maxIdx - 1] < p:
        #     maxIdx += 1

        # minReturn = min(self.minGamma, self.p)

        # return order[0:max(minReturn, min(maxIdx, cap))]
        

    #########################################################################




    #########################################################################
    ###
    ### THETA_LIKELIHOOD_AND_GRADIENT
    ###
    ### Last Updated: 4/13/16
    ###
    ### Note: Returns the likelihood and its gradient for a given value of theta.
    ###

    def theta_L_gradL_gamma(self,theta):

        # Calculate all values that are constant for every term in the sum
        expTheta = np.exp(theta * 0.5) * np.eye(self.p)
        L = expTheta.dot(self.S).dot(expTheta) + np.eye(self.p)
        eigVals, eigVecs = np.linalg.eigh(L)
        normalizer = reduce(lambda a, x: a * (x + 1.), eigVals, 1.)
        K = np.eye(self.p) - eigVecs.dot((1./(eigVals+1.)) * np.eye(self.p)).dot(eigVecs.T)
        K1 = K.dot(np.ones((self.p,1)))

        def getGamma(idx):
            gamma = np.zeros((self.p,1))
            for i in idx:
                gamma[i,0] = 1.
            return gamma

        L = 0.
        GradL = np.zeros(theta.shape)
        nothing = True
        for i in range(self.larsCap):
            for p in itertools.combinations(self.larsTopGamma[0:self.larsCap],i+1):
                gamma = getGamma(p)
                if p in self.ignore:
                    continue
                else:
                    nothing = False
                logPYGam = self.logPYconditionalGammaX(gamma)
                # pYGam = self.ALTpYconditionalGammaX(gamma)
                
                pGamTheta = self.memoizer.FdetL(gamma,theta)
                # Compute coefficient in the logspace
                k = np.exp(logPYGam + np.log(pGamTheta) + np.log(normalizer))
                if k == 0.:
                    # If it's too small, and p(y|gamma) was negligable on its own, 
                    if np.exp(logPYGam) == 0.:
                        self.ignore.add(p)

                L += k
                GradL += k * (gamma - K1)


        if nothing:
            print "HOLD UP - EVERYTHING IS IGNORED"
            self.larsCap += 1

        L *= normalizer
        GradL *= normalizer

        if L == 0.:
            GradLogL = GradL * 1e6
            L = sys.float_info.min
        else:
            GradLogL = GradL / L
            L = np.log(L)

        return L, GradLogL



    #########################################################################


    #########################################################################
    ###
    ### P_Y_CONDITIONAL_GAMMA_X
    ###
    ### Last Updated: 4/14/16
    ###
    
    def pYconditionalGammaX(self,gamma):
        # p(y|gamma)
        det = self.memoizer.FdetSLam(gamma, self.c)
        if det == 0.:
            return np.finfo(float).eps
        diffProj = self.memoizer.FDifferenceProjection(gamma,self.c)
        var = self.var
        n = self.n
        a0 =self.a0
        b0 = self.b0

        pYcondGamma = 1. / (np.sqrt(det) * (b0 + diffProj) ** (a0 + n * 0.5))
        
        # pYcondGamma = np.exp(-1 * diffProj / (2. * var)) / np.sqrt(det)
        # print pYcondGamma

        # if pYcondGamma == 0.:
            # print "Det = %s" % repr(det)
            # print "diffProj = %s" % repr(diffProj)
            # print "var = %s" % repr(var)
            # print "n = %s" % repr(n)
        return pYcondGamma

    #########################################################################


    #########################################################################
    ###
    ### LOG_P_Y_CONDITIONAL_GAMMA_X
    ###
    ### Last Updated: 4/20/16
    ###
    
    def logPYconditionalGammaX(self,gamma):
        # p(y|gamma)
        det = self.memoizer.FdetSLam(gamma, self.c)
        if det == 0.:
            return np.finfo(float).eps
        diffProj = self.memoizer.FDifferenceProjection(gamma,self.c)
        var = self.var
        n = self.n

        logPYcondGamma = -1. * diffProj / var - 0.5 * np.log(det)
        
        return logPYcondGamma

    #########################################################################


    #########################################################################
    ###
    ### ALT_P_Y_CONDITIONAL_GAMMA_X
    ###
    ### Last Updated: 4/14/16
    ###
    
    def ALTpYconditionalGammaX(self,gamma):
        # p(y|gamma)
        det = self.memoizer.FdetSLam(gamma, self.c)
        if det == 0.:
            return np.finfo(float).eps
        diffProj = self.memoizer.FDifferenceProjection(gamma,self.c)
        var = self.var
        n = self.n

        pYcondGamma = np.exp(-1. * diffProj / var) / np.sqrt(det) 
        
        return pYcondGamma

    #########################################################################



    #########################################################################
    ###
    ### DPP_SAMPLER
    ###
    ### Last Updated: 4/14/16
    ###
    
    def DPPSampler(self, eigVals, eigVecs, indices = False):

        # Build set J with elements i from [1, M] w.p. lam_i / (lam_i + 1)
        J = self.eigSample(eigVals)
        V = np.array([eigVecs[:, i] for i in J]).T
        Y = []

        # Select elements of Y, with probability as averaged sum of squares of
        # ith component of eivenvectors in V. 
        while len(V) > 0:
            N = V.shape[0]
            n = V.shape[1]
            PDF = np.array([V[i, :].T.dot(V[i,:]) / float(n) for i in range(0,N)])
            pDist = np.cumsum(PDF)

            r = random.random()
            i = len([x for x in pDist if x < r])
            Y.append(i)
            V = self.orthogonalize(V, i)


        # Return Y as a numpy array column vector
        if indices:
            return np.array([Y]).T
        else:
            Y_Lspace = np.zeros((self.p,1))
            for i in range(0,len(Y)):
                Y_Lspace[Y[i]] = 1.0
            return Y_Lspace

    #########################################################################


    #########################################################################
    ###
    ### ORTHOGONALIZE
    ###
    ### Last updated: 11/03/15
    ###
    ### Note: Given a set of R basis vectors and i \in [0, N-1], compute a 
    ###       new set of R-1 vectors that are orthogonal to e_i, but span the
    ###       rest of the original space spanned by the R original vectors. 
    ###

    def orthogonalize(self, basis, i):
        """
         Params: basis is an orthonormal basis for some k-dimensional subspace
                 of R^N. i indicates the dimension within N to which the 
                 produced basis should be orthogonal. 
         Result: a new set of basis vectors, orthogonal to e_i, but otherwise
                 spanning the original space. 
        """

        # Check type of eigVals
        assert type(basis) == np.ndarray

        N = basis.shape[0]
        d = basis.shape[1]
        dimNull = N - d + 1 # dimension of null space of newBasis

        if d == 1:
            return np.array([])

        eps = 1.0e-8 # Don't want to be dividing by some very small number
        # Get first column of basis with nonzero entry in ith row
        j = 0 # column number
        while j < d:
            if abs(basis[i][j]) < eps:
                j += 1
            else:
                break
        
        # kth column minus the (j) scaling column times ratio of ith element in kth to jth
        newOrthogonal = np.array([basis[:,k] - basis[:,j] * (float(basis[i,k]) / basis [i,j]) for k in range(0,basis.shape[1]) if k != j]).T
        newOrthonormal, R = linalg.qr(newOrthogonal)


        # Since we reduced the span of the basis, QR should have a larger nullspace (dim = n), given
        # by n rows of zeros in R. In Python's implementation of QR decomposition, these should
        # appear in the final n rows. Verify that this is the case. We verify this when we remove the
        # corresponding n rightmost columns of Q
        for nullCheck in range(0, dimNull):
            assert sum(R[N - nullCheck - 1, :]) == 0
        V = np.array(newOrthonormal[:,0:N - dimNull])

        return V


    #########################################################################


    #########################################################################
    ###
    ### EIG_SAMPLE
    ###
    ### Last updated: 11/02/15
    ###
    ### Note: Given a set of values k_i, we return a set of indices, each 
    ###       index i is included in the set w.p. k_i / (k_i + 1)
    ###

    def eigSample(self, eigVals):
        """
         Params: eigVals is a set of eigenValues.   
         Result: a set of indices, each index i is included in the set 
                 w.p. lam_i / (lam_i + 1)
        """

        # Check type of eigVals
        assert type(eigVals) == np.ndarray

        M = len(eigVals)

        # convert eigVals to probabilities
        probs = np.array([lam / (1.0 + lam) for lam in eigVals])

        # Create inclusion vector
        indices = np.array([i for i in range(0, M) if probs[i] > random.random()])

        return indices
        
    # def testEigSample(N):
    #     eigVals = np.array(random.rand(20,1))
    #     freq = np.zeros(eigVals.shape)
    #     for i in range(0,N):
    #         indices = eigSample(eigVals)
    #         for j in range(0, len(indices)):
    #             freq[indices[j]] += 1.0/N
    #     probs = np.array([lam / (1.0 + lam) for lam in eigVals])
    #     print probs - freq
        

    #########################################################################



    #########################################################################
    ###
    ### PREDICT
    ###
    ### Last Updated: 4/16/16
    ###

    def predict(self, X_test):
        beta = self.beta
        predictions = X_test.dot(beta)
        return predictions


    #########################################################################




if __name__ == '__main__':

    n = 400

    fn = "StressTestResults%s.txt" % datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')

    for features in range(6,100):
        if features < 10:
            sparse = .6
        elif features < 35:
            sparse = .8
        else:
            sparse = .9

        dataGen = CDG.CollinearDataGenerator(p=features, sparsity=sparse)
        print "Gamma Star: %s" % repr(dataGen.gamma)
        X = dataGen.getX(n)
        y = dataGen.getY(X)
        lezzgo = PO(X,y,max_T = 500, GA_max_T = 500)
        with open(fn,'a') as f:
            f.write('With p = %d, sparsity = %f\n' % (features, sparse))
            f.write('   my theta: %s\n' % repr(lezzgo.theta))
            f.write('   my gamma: %s\n' % repr(lezzgo.gamma))
            f.write('   gamma star: %s\n' % repr(dataGen.gamma))
            f.write('\n-----------------------\n')


    # dataGen = CDG.CollinearDataGenerator(p=12, sparsity = .8)
    # print "Gamma Star: %s" % repr(dataGen.gamma)
    # X = dataGen.getX(n)
    # y = dataGen.getY(X)
    # lezzgo = PO(X,y)
    # with open(fn,'a') as f:
    #     f.write('With p = 12, collinear_p = 0.5\n')
    #     f.write('   my theta: %s\n' % repr(lezzgo.theta))
    #     f.write('   my gamma: %s\n' % repr(lezzgo.gamma))
    #     f.write('   gamma star: %s\n' % repr(dataGen.gamma))
    #     f.write('\n-----------------------\n')


    # dataGen = CDG.CollinearDataGenerator(p=20, sparsity = .8)
    # print "Gamma Star: %s" % repr(dataGen.gamma)
    # X = dataGen.getX(n)
    # y = dataGen.getY(X)
    # lezzgo = PO(X,y)

    # with open(fn,'a') as f:
    #     f.write('With p = 20, collinear_p = 0.5\n')
    #     f.write('   my theta: %s\n' % repr(lezzgo.theta))
    #     f.write('   my gamma: %s\n' % repr(lezzgo.gamma))
    #     f.write('   gamma star: %s\n' % repr(dataGen.gamma))
    #     f.write('\n-----------------------\n')

    # n = 400
    # X = KKData.genX(n=n)
    # p = X.shape[1]
    # y = KKData.genY(X)

    # lezzgo = PO(X,y)
