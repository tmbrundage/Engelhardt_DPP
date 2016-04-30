function pv=pastVals(N)
assert(N>=1);
pv.i = 0;
pv.N = N;
pv.v = zeros(N,1);
pv = class(pv, 'pastVals');
