% ONE(N,I) return a vector of N 0s with a 1 at position I.
function v = one(N,i)
v=zeros(N,1);
v(i) = 1;