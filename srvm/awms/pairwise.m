% PAIRWISE(F,V)  Perform F(V(I),V(I+1)) for all I and return the result.
function p = pairwise(f, v);
p = nan(length(v)-1,1);
for i = 1:length(v)-1
  p(i)=f(v(i+1), v(i));
end
  