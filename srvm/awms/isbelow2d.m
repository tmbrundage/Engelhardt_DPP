% ISBELOW2D Test if something is < 2 dimensional.
% 
% Matlab's handling of dimensionality is minbogglingly braindamaged. this
% function tries to figure out whether something *presumably* is a 1D or
% scalar array/cell/struct
function bool=isbelow2d(x)
bool = ndims(x) == 2 && min(size(x)) <= 1;
% TEST
% $$$ >> isbelow2d([])
% $$$ 
% $$$ ans =
% $$$ 
% $$$      1
% $$$ 
% $$$ >> isbelow2d(1)
% $$$ 
% $$$ ans =
% $$$ 
% $$$      1
% $$$ 
% $$$ >> isbelow2d([1,2,3])
% $$$ 
% $$$ ans =
% $$$ 
% $$$      1
% $$$ 
% $$$ >> isbelow2d([1,2,3]')
% $$$ 
% $$$ ans =
% $$$ 
% $$$      1
% $$$ 
% $$$ >> isbelow2d([1,2,3;4,5,6]')
% $$$ 
% $$$ ans =
% $$$ 
% $$$      0
