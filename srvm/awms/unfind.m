function [bool] = unfind(ind,N)
bool = logical(zeros(N,1));
for i=ind, bool(i) = true; end;
