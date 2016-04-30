function assertHelper(k, exp, msg, varargin)
global ASSERTIONS_DONT_PRODUCE_ERRORS
if size(ASSERTIONS_DONT_PRODUCE_ERRORS) == 0, 
  ASSERTIONS_DONT_PRODUCE_ERRORS=false;
end
if     nargin < 3, msg = 'Assertion failed'; 
else               msg = sprintf(msg, varargin{:}); end
if ~exp && ASSERTIONS_DONT_PRODUCE_ERRORS~=true
  k(msg);
end