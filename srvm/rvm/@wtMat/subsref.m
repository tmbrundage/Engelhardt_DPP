function X2=subsref(X,sel)
if strcmp(sel.type, '.'),
  X2=subsref(struct(X),sel);
else
  if ~strcmp(X.sel, ':'), 
    error('Currently wtMats can only be indexed once');
  end
% $$$   disp(sel);
  if ~(strcmp(sel.type, '()') && length(sel.subs) == 2 && ...
       strcmp(sel.subs{1+X.transposed},':'))
    error('Indexing must look like so: X(:,sel)');
  end
  % special case indexing of single column to yield normal vector
  if length(sel.subs{2}) == 1
    v = zeros(size(X,1),1); v(sel.subs{2}) = 1;
    X2 = X * v;
  else
    X2 = X;
    X2.sel = sel.subs{2-X.transposed};
  end
end
