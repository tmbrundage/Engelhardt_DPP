import numpy as np

n = 250
p = 25

X = np.random.randn(n,p)
S = X.T.dot(X)

dist = 0.00001
eps = dist * np.ones((p,1))

def expDiag(theta):
    out = np.zeros((p,p))
    for i in range(p):
        out[i,i] = np.exp(theta[i,0] * 0.5)
    return out

def getProdEye(theta):
    expTheta = expDiag(theta)
    prod = expTheta.dot(S).dot(expTheta) + np.eye(p)
    return prod

def getVal(theta):
    return np.linalg.det(getProdEye(theta))

# def H1(theta):
#     M = getProdEye(theta)
#     coeff = 2.0 * np.linalg.det(M)
#     mats = np.linalg.inv(M).dot(expDiag(theta)).dot(S).dot(np.ones((p,1)))
#     return coeff * mats

def H(theta):
    M = getProdEye(theta)
    coeff = np.linalg.det(M)
    DIAG = expDiag(theta)
    mats = np.linalg.inv(M).dot(DIAG).dot(S).dot(DIAG).dot(np.ones((p,1)))
    return coeff * mats


# def test(tau):
#     theta = np.random.randn(p,1)
#     h1 = H1(theta)
#     h2 = H2(theta)
#     v0 = getVal(theta)
#     v1 = getVal(theta + eps)
    
#     est1 = v0 + eps.T.dot(h1)
#     est2 = v0 + eps.T.dot(h2)
#     stupid = v0 + eps.T.dot(np.ones((p,1)))

#     print v1
#     print est1
#     print est2
#     print stupid
    # del1 = (est1-v1)/v1
    # del2 = (est2-v1)/v1


    # return (abs(del1)<tau, abs(del2)<tau, del1>0, del2>0)


def walk():
    nSteps = 1e5
    theta0 = np.random.randn(p,1)
    thetaT = np.random.randn(p,1)
    print thetaT-theta0

    delta = (thetaT - theta0) / nSteps
    current = getVal(theta0)
    theta = theta0
    for i in range(int(nSteps)):
        current += delta.T.dot(H(theta))
        theta += delta

    actual = getVal(thetaT)
    error = (current - actual) / actual
    return error


if __name__ == '__main__':
    over = 0
    under = 0
    big = 0
    tau = 0.005
    N = 10000
    fname = 'DiffDetTest.txt'
    for i in range(N):
        error = walk()
        if error > 0:
            over += 1
        else:
            under += 1
        if abs(error) > tau:
            big += 1
        if i % 100 == 0:
            with open(fname,'a') as f:
                f.write("At step %d\n"%i)
                f.write("    %d over, %d under\n" % (over, under))
                f.write("    %d over %f\n" % (big,tau))


