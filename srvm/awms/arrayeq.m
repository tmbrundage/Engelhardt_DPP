% ARRAYEQ(A,B) is true iff A and B have the same shape and contents.
% handles NaNs (i.e. ARRAYEQ([NaN,1,2], [NaN,1,2]) is true).
function truth = arrayeq(a,b)
truth = ~(any(size(a) ~= size(b)) || (any(a(:) ~= b(:)) && ~any(isnan(a(:)))) ...
          || any(isnan(a(:)) ~= isnan(b(:))) || any(a(~isnan(a(:))) ~= ...
                                                  b(~isnan(a(:)))));

% $$$ >> arrayeq([NaN,1,2], [NaN,1,2])
% $$$ 
% $$$ ans =
% $$$ 
% $$$      1
% $$$ 
% $$$ >> arrayeq([NaN,1,2], [NaN,1])
% $$$ 
% $$$ ans =
% $$$ 
% $$$      0
% $$$ 
% $$$ >> arrayeq([NaN,1,2], [NaN,1,3])
% $$$ 
% $$$ ans =
% $$$ 
% $$$      0
% $$$ 
% $$$ >> arrayeq([1,2;3,4], [1,2,3,4])
% $$$ 
% $$$ ans =
% $$$ 
% $$$      0
% $$$ 
% $$$ >> arrayeq([1,2;3,4], [1,2;3,4])
% $$$ 
% $$$ ans =
% $$$ 
% $$$      1
