% CELLEQ(S,T) compare cells S and T for equality.
% See also GENEQ
function bool=celleq(s,t)
assert(strcmp(class(s),'cell') && strcmp(class(t),'cell'));
assert(isbelow2d(s) && isbelow2d(t), 'I''m currently not masochistic enough for >1D');
bool = false;
if length(s) ~= length(t)
  return
else
  for i=1:length(s)
    if ~geneq(s{i}, t{i}), return; end
  end
  bool = true;
end

% $$$ >> celleq({},{})
% $$$ 
% $$$ ans =
% $$$ 
% $$$      1
% $$$ 
% $$$ >> celleq({1},{1})
% $$$ 
% $$$ ans =
% $$$ 
% $$$      1
% $$$ 
% $$$ >> celleq({1,2},{1})
% $$$ 
% $$$ ans =
% $$$ 
% $$$      0
% $$$ 
% $$$ >> celleq({1,2},{1,2})
% $$$ 
% $$$ ans =
% $$$ 
% $$$      1
% $$$ 
% $$$ >> celleq({1,2},{1,3})
% $$$ 
% $$$ ans =
% $$$ 
% $$$      0
% $$$ >> celleq({'a'}, {'a'})
% $$$ 
% $$$ ans =
% $$$ 
% $$$      1
% $$$ 
% $$$ >> celleq({'a'}, {'ab'})
% $$$ 
% $$$ ans =
% $$$ 
% $$$      0
% $$$ >> celleq({'a', int32(1)}, {'a',1})
% $$$ 
% $$$ ans =
% $$$ 
% $$$      0
% $$$ >> celleq({1}, 1)xo
% $$$ 
% $$$ ans =
% $$$ 
% $$$      0
