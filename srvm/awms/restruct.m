% $$$ restruct(struct('a',3, 'b', 4), 'c',5, 'b',40)
% $$$ 
% $$$ ans = 
% $$$ 
% $$$     a: 3
% $$$     b: 40
% $$$     c: 5
function s = restruct(s,varargin)
assert((length(varargin) ./ 2) == floor(length(varargin) ./ 2))
for i=0:(length(varargin)-1) ./ 2
  s=setfield(s, varargin{1+2.*i}, varargin{1+2.*i+1}); 
end
