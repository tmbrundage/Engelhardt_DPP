% ARGMAX Index of largest component
%   I=ARGMAX(A) is identical to [dummy,I] = MAX(A).
function I = argmax(varargin);
[dummy,I] = max(varargin{:});