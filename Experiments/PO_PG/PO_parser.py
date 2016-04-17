import numpy as np
import re
import matplotlib.pyplot as plt

files = ['StressTestResults_500_20160415_183351.txt',
         'StressTestResults_1000_20160415_183526.txt',
         'StressTestResults_500_1000_20160415_190017.txt',
         'StressTestResults_1000_500_20160415_190235.txt']
saves = ['183351','183526','190017','190235']
if __name__ == '__main__':
    i = 0
    fn = files[i]
    with open(fn,'r') as content_file:
        content = content_file.read()
    tests = content.split('\n-----------------------\n')
    # print len(tests)
    # print tests[0]
    # theta = re.compile(r'my\stheta:\sarray\((.*)\)\smy')
    # theta = re.compile(r'my\stheta:\sarray\((.*)\)\n\s*my',re.DOTALL)
    gamma = re.compile(r'my\sgamma:\sarray\((.*)\)\n\s*gamma',re.DOTALL)
    gammaStar = re.compile(r'gamma\sstar:\sarray\((.*)\)',re.DOTALL)


    errors = []
    xs = []
    for s in tests:
        found = gamma.search(s)
        if not found:
            break
        myGamma = found.group(1).split(',\n       ')
        myGamma = np.array([[float(x.strip('[').strip(']')) for x in myGamma]]).T

        anGamma = gammaStar.search(s).group(1).split(',')
        anGamma = np.array([[float(x.strip('[').strip(']')) for x in anGamma]]).T
        err = float(sum(myGamma)) / sum(anGamma)
        errors.append(float(err))
        xs.append(len(myGamma))
        # print "For %d features, error = %d: %f" % (len(myGamma),err, float(err)/len(myGamma))

    print errors
    plt.hist(errors, 20)
    plt.savefig('%s_ratios_hist.png' % saves[i])

    # plt.plot(xs,errors)
    # plt.show()
    # b = gammaStar.search(tests[0])   
    # print b.group(1)
