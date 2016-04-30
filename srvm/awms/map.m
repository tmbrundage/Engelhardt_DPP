% MAP(F,V,[W]) Map a function over one or two arrays or cells.
%
%     >> [map(@(x) -x, [1,2,3]); map(@(x,y) x-y, [1,2,3],[1,2,3])]
%     
%     ans =
%     
%         -1    -2    -3
%          0     0     0

function fa = map(f, a, b)
N = length(a);
fa = zeros(size(a));
%FIXME make this work around matlab's braindamages properly
if strcmp(class(a), 'cell')
  if nargin == 2
    for i=[1:N],  fa(i) = f(a{i}); end
  else
    assert(length(b) == N);
    for i=[1:N],  fa(i) = f(a{i}, b{i}); end
  end
else 
  if nargin == 2
    for i=[1:N],  fa(i) = f(a(i)); end
  else
    assert(length(b) == N);
    for i=[1:N],  fa(i) = f(a(i), b(i)); end
  end
end