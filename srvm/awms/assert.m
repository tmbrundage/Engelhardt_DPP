% ASSERT Simulate assertions in matlab 
%
%   ASSERT(EXPRESSION) will raise an error
%   if EXPRESSION does not evaluate to TRUE unless the global
%   ASSERTIONS_DONT_PRODUCE_ERRORS is set to TRUE.
function assert(exp, varargin)
assertHelper(@error, exp, varargin{:});

