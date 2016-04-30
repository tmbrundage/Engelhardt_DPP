% MYGETFIELD(A,F,DEFAULT)  Like GETFIELD, but accepts a DEFAULT VALUE.
%
%   If ~ISFIELD(A,F), then DEFAULT is returned, unless DEFAULT is a
%   function handle in which case DEFAULT() is returned.
function res = mygetfield(s, field, default);
if nargin==2,default = nan; end
if isfield(s, field)
  res = getfield(s,field);
else
  if strcmp(class(default), 'function_handle')
    res = feval(default);
  else
    res = default;
  end
end