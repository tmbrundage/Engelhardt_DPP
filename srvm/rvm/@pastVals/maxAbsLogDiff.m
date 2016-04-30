function delta = maxAbsLogDiff(pv)
if pv.i < pv.N,
  delta = inf;
else
  delta = max(abs(log(pv.v(1:end-1))-log(pv.v(end))));
end