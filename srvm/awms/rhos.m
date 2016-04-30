function d = rhos(v)
d = pairwise(@(a,b)a./b, v);