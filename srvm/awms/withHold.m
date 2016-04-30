function withHold(f)
holdState = ishold;
hold on;
try
  f();
catch
  if ~holdState, hold off;end
  rethrow(lasterror)
end
if ~holdState, hold off;end  
