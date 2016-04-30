% CLIP(A, H, L)  set all values A(I) to be L<=A(I)<=H
function x=clip(x,L,H)
x(x>H) = H; x(x<L)=L;


