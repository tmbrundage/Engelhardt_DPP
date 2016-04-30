% GENEQ  generic equality function
%
%    GENEQ(A,B) returns whether arbitrary objects A and B (possibly of
%    different classes, in which case the result is false) are identical. The
%    result is always scalar true or false.
%    
%     >> 1==ones(1,1,'int32')
%     
%     ans =
%     
%          1
%     
%     >> geneq(1,ones(1,1,'int32'))
%     
%     ans =
%     
%          0

function truth = geneq(a,b)
if ~strcmp(class(a),class(b)), 
  truth=false;
else
  switch class(a)
   case 'struct'
    truth=structeq(a,b);
   case 'cell'
    truth=celleq(a,b);
   case 'string'
    truth=strcmp(a,b);
   otherwise 
    truth=arrayeq(a,b);
  end
end
% $$$ >> geneq('a', 'a')
% $$$ 
% $$$ ans =
% $$$ 
% $$$      1
% $$$ 
% $$$ >> geneq('a', 'b')
% $$$ 
% $$$ ans =
% $$$ 
% $$$      0
% $$$ 
% $$$ >> geneq('a', 'ab')
% $$$ 
% $$$ ans =
% $$$ 
% $$$      0
% $$$ 
% $$$ >> geneq('a', [1,2,3])
% $$$ 
% $$$ ans =
% $$$ 
% $$$      0
% $$$ 
% $$$ >> geneq('a', struct('a','a'))
% $$$
% $$$ ans =
% $$$ 
% $$$      0
