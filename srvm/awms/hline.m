function hline(z, style)
% hline(z, style)
%          Plot the x axis on the current graph.
%
% z        Plot 1 or more horizontal line(s) at height(s) z (default: 0)
% style    Linestyle.

% Copyright (c)   Richard Everson,	Imperial College 1998
% $Id: hline.m,v 1.2 2006/03/08 10:19:21 aschmolc Exp $
% hacked by a.schmolck
if nargin < 1 | isempty(z), z = 0.0; end
if nargin < 2 | isempty(style), style = ':'; end
holdState = ishold;
hold on;
v = axis;
for i=1:length(z)
  plot(v(1:2), [z(i), z(i)], style);
end
if ~holdState, hold off;end
return
