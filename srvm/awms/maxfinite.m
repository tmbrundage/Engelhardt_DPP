function m = maxfinite(a)
a(~isfinite(a)) = -inf;
m=max(a);