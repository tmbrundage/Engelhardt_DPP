% DEMEANED(X,DIM) Removes the mean along dimension DIM (default=1)
% 
% >> demeaned([1,2,3])
%    -1     0     1
% >> demeaned([1,2,3;4,5,6],1)
% 
% ans =
% 
%    -1.5000   -1.5000   -1.5000
%     1.5000    1.5000    1.5000
% 
% >> demeaned([1,2,3;4,5,6],2)
% 
% ans =
% 
%     -1     0     1
%     -1     0     1
function X=demeaned(X,dim);
if is1d(X) && nargin == 1
  X=X-mean(X);
else
  if nargin == 1
    dim = 1; 
  end
  if dim == 1
    X=X-repmat(mean(X,dim),size(X,1), 1);
  elseif dim == 2
    X=X-repmat(mean(X,dim),1, size(X,2));
  else
    error('only works for dim <= 2')
  end
end