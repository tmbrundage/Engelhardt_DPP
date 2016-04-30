% I=ARGSORT(A)  Return indexes I so that I=ARGSORT(A); B=A(I) is like B=SORT(A).
function I = argsort(varargin);
[dummy,I] = sort(varargin{:});