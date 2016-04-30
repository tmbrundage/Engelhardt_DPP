function [pv] = add(pv, x)
pv.i = min(pv.N, pv.i+1);
pv.v(1:end-1) = pv.v(2:end);
pv.v(end) = x;
