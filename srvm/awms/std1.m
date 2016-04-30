%FIXME how sensible is this function?
function X1 = std1(X)
sigma = std(X);
X1=X/sigma;
