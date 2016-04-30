function [dif] = maxdif(a,b, epsilon)
if nargin == 2, epsilon = 1E-10; end
dif = clean(max(abs(a(:)-b(:))));
  