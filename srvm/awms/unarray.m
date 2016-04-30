function varargout = unarray(a)
for i=1:length(a)
  varargout{i} = a(i);
end