#########################################################################
####  Ted Brundage, M.S.E., Princeton University
####  Advisor: Barbara Engelhardt
####
####  Code: Bayesian Network Variable
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

import abc
from collections import deque

#########################################################################


#########################################################################
###
### ERROR TYPES
###

class StateError(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class ValueError(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class ArgumentError(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

#########################################################################


class BNV(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def isiterative(self):
        return 'should never get here, BNV.isiterative'

    @abc.abstractproperty
    def defaultAlpha(self):
        return 'should never get here, BNV.defaultalpha'

    @abc.abstractproperty
    def defaultThreshold(self):
        return 'should never get here, BNV.defaultThreshold'

    @abc.abstractmethod
    def defaultValue(self, *args):
        return 'should never get here, BNV.defaultvalue'


    def __init__(self,**kwargs):
        self.val = self.defaultValue() # defalut value 
        if self.isiterative:
            self.alpha = self.defaultAlpha # Default learning rate
        param_b = False
        set_b = False
        self.n_converge = 10
        for name,value in kwargs.items():
            if name == 'val':
                self.val = value
                set_b = True
            elif name == 'param':
                param = value
                param_b = True
            elif name == 'n_converge':
                self.n_converge = int(value)
            elif name == 'alpha':
                if self.isiterative:
                    self.alpha = float(value)
                else:
                    print 'Warning: no alpha parameter for non-iterative BNV.'
        # If both a value and a parameter have been given,
        # prioritize the value. Otherwise, give default 
        # based on the parameter.
        if param_b and not set_b:
            self.val = self.defaultValue(param)

        self.recentVals = deque()
        self.skipped = 0

    def isconverged(self):
        if len(self.recentVals) >= self.n_converge:
            for elem in self.recentVals:
                if elem > self.defaultThreshold:
                    return False
            return True
        else:
            return False

    def val_getter(self):
        return self.val

    def val_setter(self, state, newval):
        if self.check(state,newval):
            self.val = newval
        else:
            print "newval: "
            print newval
            
        # else:
            # print "newval: "
            # print newval
            # print "Invalid new value for %s" % self.__class__.__name__
            # raise ValueError("Invalid new value for %s" % self.__class__.__name__)

    def check_BNVs(self, state, reqKeys):
        for key in reqKeys:
            if key not in state.bnv:
                print 'State includes %d BNVs:' % len(state.bnv)
                for k in state.bnv.keys():
                    print k
                raise StateError('Variable %s not found in current state.' % key)

    def check_HPs(self, state, reqParams):
        for key in reqParams:
            if key not in state.hp:
                print 'State includes %d hyperparameters:' % len(state.hp)
                for k in state.hp.keys():
                    print k
                raise StateError('Hyperparameter %s  not found in current state.' % key)

    @abc.abstractmethod
    def likelihood(self, state):
        """
            Calculate the likelihood of the current value of this variable.
            State is a dictionary with all relevant values of other variables
            for this calculation.
        """
        return

    @abc.abstractmethod
    def gradLikelihood(self, state):
        """
            Calculate the gradient of the likelihood of the current value 
            of this variable. State is a dictionary with all relevant values 
            of other variables for this calculation.
            Note, not all variables necessarily have defined gradients. 
        """
        return

    @abc.abstractmethod
    def update(self, state):
        """
            Actually updates the current state of this variable. This will
            use the gradLikelihood function for variables with defined
            likelihood gradients, but can also perform other calculations
            like greedy algorithms for those with different features. State
            again gives the curent state of all other relevant variables for 
            this calculation. 
        """
        return

    @abc.abstractmethod
    def check(self, state, val):
        """
            Check that this value satisfies all constraints given by the current state
            and value. 
        """
        return

    def updateChanges(self,val):
        if len(self.recentVals) >= self.n_converge:
            self.recentVals.popleft()
            self.recentVals.append(val)
        else:
            self.recentVals.append(val)

    def clearHistory(self):
        self.recentVals = deque()
        self.skipped = 0
