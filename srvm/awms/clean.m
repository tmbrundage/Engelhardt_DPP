% CLEAN  Set values close to 0 to 0.
%
%   CLEAN(A) = CLEAN(A, 1E-10) set A(abs(A)<=1E-10) = 0
function [m] = clean(m, epsilon)
if nargin < 2 epsilon =1E-10; end
m(abs(m)<=epsilon) = 0;

