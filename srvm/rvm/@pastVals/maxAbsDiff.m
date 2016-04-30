function delta = maxAbsDiff(pv)
if pv.i < pv.N,
  delta = inf;
else
  delta = max(abs(pv.v(1:end-1)-pv.v(end)));
end