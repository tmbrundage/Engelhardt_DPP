function n = sqNorm(XX, dim)
if nargin == 1, dim = 1; end
n = sum(XX.^2, dim);
