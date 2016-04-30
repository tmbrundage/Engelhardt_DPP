% NORMED  Return argument divided by its norm.
function X=normed(X,varargin);
X=X./norm(X,varargin{:});