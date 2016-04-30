% DELTAS(V)  compute pairwise differences in V.
function d = deltas(v)
d = pairwise(@(a,b)a-b, v);