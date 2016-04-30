% KASSERT  Like ASSERT but calls KEYBOARD rather than ERROR.
function kassert(exp, varargin)
assertHelper(@helper, exp, varargin{:});
function helper(msg)
display(msg);
keyboard;
