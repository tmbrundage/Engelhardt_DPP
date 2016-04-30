% ARGMIN Index of smallest component
%   I=ARGMIN(A) is identical to [dummy,I] = MIN(A).
function I = argmin(varargin);
[dummy,I] = min(varargin{:});