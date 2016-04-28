import numpy as np 

try:
    np.log(-3.)
    print "STILL CONTINUED"
except:
    print "Skipped to except"