% STRUCTEQ(S,T) compare structures S and T for equality.
%
%  See also: GENEQ
function bool=structeq(s,t)
assert(isbelow2d(s) && isbelow2d(t), 'I''m currently not masochistic enough for >1D');
bool = false;
assert(strcmp(class(s),'struct') && strcmp(class(t),'struct'));
fs=fieldnames(s);
if ~celleq(fs, fieldnames(t))
  return;
else
  for i=1:length(fs)
    if ~geneq(getfield(s,fs{i}), getfield(t, fs{i}))  return; end
  end
  bool=true;
end

  
% TEST 
% $$$ >> structeq(struct('ab','a'), struct('a', 'a'))
% $$$ 
% $$$ ans =
% $$$ 
% $$$      0
% $$$ 
% $$$ >> structeq(struct(), struct())
% $$$ 
% $$$ ans =
% $$$ 
% $$$      1
% $$$ 
% $$$ >> structeq(struct('a',1), struct('a',1))
% $$$ 
% $$$ ans =
% $$$ 
% $$$      1
% $$$ 
% $$$ >> structeq(struct('a',1), struct('a',2))
% $$$ 
% $$$ ans =
% $$$ 
% $$$      0
% $$$ 
% $$$ >> structeq(struct('a',1), struct('b',1))
% $$$ 
% $$$ ans =
% $$$ 
% $$$      0
% $$$ 
% $$$ >> structeq(struct('a',1,'b',2), struct('a',1))
% $$$ 
% $$$ ans =
% $$$ 
% $$$      0
% $$$ 
% $$$ >> structeq(struct('a',1),struct('a',1,'b',2))
% $$$ 
% $$$ ans =
% $$$ 
% $$$      0
% $$$ 
% $$$ >> structeq(struct('a',1,'b',2),struct('a',1,'b',2))
% $$$ 
% $$$ ans =
% $$$ 
% $$$      1
