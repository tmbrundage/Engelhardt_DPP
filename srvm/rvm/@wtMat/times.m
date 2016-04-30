function res = times(B,X)
% HACK this is only meant to work for calculateFullLikelihood
% specifically the form must be [a,a,a...;b,b,b...;c,c,c...;...].*X'
% but the identity of all row entries is not checked
[M,N]=size(X);
assert(isscalar(B) || ...
       (X.transposed && all(size(X)==size(B)) && B(1,1) == B(1,end)));
if isscalar(B), %XXX this is rather dumb
  if ~X.transposed, 
    res = (B.*X')'; return; 
  else
    B = repmat(B,M,1); 
  end; %XXX
end

res = zeros(M,N);
tmp = [1:N];
reindex = tmp(X.sel);
for i = 1:M
  res(i,reindex(i)) = B(i,1);
  res(i,:) = IWT_PO(res(i,:), X.L, X.filt);
end
