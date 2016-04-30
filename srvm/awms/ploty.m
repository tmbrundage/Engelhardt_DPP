% PLOTY  Convenience wrapper around PLOT
% PLOTY(Y1,'g',Y2,'r+',Y3) == 
% PLOT([1:length(Y1), 'g',[1:length(Y1)], Y2, 'r+',[1:length(Y1)],Y3))
function handles = ploty(y,varargin)
if is1d(y), 
  x=[1:length(y)]'; 
else 
  x=size(y,1);
end
newargs = {x,y};
for y_i = varargin
  y_i=y_i{1};
  switch class(y_i)
   case {'char'}
    newargs{end+1} = y_i;
   otherwise
    newargs{end+1} = x;
    newargs{end+1} = y_i;    
  end
end
handles=plot(newargs{:});
