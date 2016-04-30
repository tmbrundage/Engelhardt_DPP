function [s,varargout] = size(X,dim)
if strcmp(X.sel,':'),
  N=X.N;
elseif islogical(X.sel)
  N=sum(X.sel);
else
  N=length(X.sel); 
end
M = X.N;
if X.transposed, tmp=M;M=N;N=tmp; end;
if nargin ~= 1,
  assert(nargout<=1, 'too many outputs');
  assert(dim == 1 || dim ==2, 'there are just 2 dimensions');
  if dim == 1, s=M; else s=N; end
  return
else
  %FIXME no idea how to emulate eye(size([1,2,3;3,4,6]))
  if nargout > 1
    s = M; varargout(1) = {N};
  else
    s = [M,N];
  end
end