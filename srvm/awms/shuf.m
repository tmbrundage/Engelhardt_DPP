% shuf(x,dim) shuffle a vector or matrix along `dim` (the default dim=1 means
% shuffle the rows). this isn't technically a "perfect shuffle" but should be
% good enough
function [x, order] = shuf(x,dim)
if nargin == 1, dim = 1; end
assert(ndims(x)<3);
order = argsort(rand(1,size(x,dim)));
if dim == 1
  x = x(order,:);
  order = order';
else
  x = x(:,order);
end