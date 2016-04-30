% IS1D Test if something is 1 dimensional.
%
%   Matlab's handling of dimensionality is minbogglingly braindamaged.
%   this function tries to figure out whether something *presumably* is a 1D
%   array/cell/struct

function bool=is1d(x)
bool = ndims(x) == 2 && min(size(x)) == 1 && max(size(x)) > 1;
% TEST
% $$$ >> is1d([1,2,3])
% $$$ is1d([1,2,3])
% $$$ 
% $$$ ans =
% $$$ 
% $$$      1
% $$$ 
% $$$ >> is1d([1,2,3]')
% $$$ is1d([1,2,3]')
% $$$ 
% $$$ ans =
% $$$ 
% $$$      1
% $$$ 
% $$$ >> is1d([1,2,3; 4,5,6]')
% $$$ 
% $$$ ans =
% $$$ 
% $$$      0
% $$$ >> is1d(1)
% $$$ 
% $$$ ans =
% $$$ 
% $$$      0
% $$$ 
% $$$ >> is1d([])
% $$$ 
% $$$ ans =
% $$$ 
% $$$      0
