function res = mtimes(X,x)
if (size(x,2)~=1), error 'foobar'; end
if ~size(x,2)==1
  error untested
  res = zeros(size(X,1), size(x,2))
  for i=1:size(res,2)
    res(:,i) = X*x(:,i);
  end
else
  assert(size(x,2)==1,'only column vector arguments currently supported');
  if X.transposed
    assert(length(x)==X.N);
    res=FWT_PO(x, X.L, X.filt);
    res=res(X.sel);
  else
    y=zeros(X.N,1);
    if strcmp(X.sel,':'),
      N=X.N;
    elseif islogical(X.sel)
      N=sum(X.sel);
    else
      N=length(X.sel); 
    end
    if N~=size(x,1), error('size mismatch'); end
    y(X.sel)=x;
    res=IWT_PO(y, X.L, X.filt);
  end
end