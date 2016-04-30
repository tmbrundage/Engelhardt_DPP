% AREF Indexing a vector or matrix.
%   Indexing as function call to work around matlab's syntactic idiocy
%   e.g. PLOT(AREF(P*Q,':',100)) is like TMP=P*Q; PLOT(TMP(:,100))
%   AREF(V,-1) == V(end); AREF(V,-2) == V(end-1) etc.
function x=aref(a,i,j)
if nargin==2
  if i < 0
    x = a(end+i+1);
  else
    x = a(i);
  end
else
  if i < 0 
    if j < 0
      x = a(end+i+1, end+j+1);
    else
      x = a(end+i+1, j);
    end
  else
    if j < 0
      x = a(i, end+j+1);
    else
      x = a(i, j);
    end
  end
end
    