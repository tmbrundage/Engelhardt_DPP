function res=withminmax(X,L,H);
if nargin == 1, L=0;H=1;end
if nargin == 2, L=0;H=L;end
res=X-min(X);
res=L+res.*((H-L)./max(abs(res)));
  