function vline(x, style)
% vline(x, style)
%          Plot the y axis on the current graph.
%
% x        Plot 1 or more vertical line(s) at x (default 0)
% style    Linestyle.

% Copyright (c)   Richard Everson,	Imperial College 1998
% modified for vectors by A.Schmolck 2005-08-09
% $Id: vline.m,v 1.2 2006/03/08 10:19:21 aschmolc Exp $
if nargin < 1 | isempty(x), x = 0.0; end
if nargin < 2 | isempty(style), style = 'b:'; end
holdState = ishold;
hold on;
v = axis;
v = v(3:4)';
N=length(x);
x_p = repmat(x(:)',2,1);
v_p = repmat(v,1,N);
plot(x_p, v_p, style);
if ~holdState, hold off;end
return
