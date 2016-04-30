% RANDRANGE(LO,HI) return a random integer  in [lo,hi) 
% RANDRANGE(LO,HI, M, N) return an M,N of array random integers in [lo,hi) 
%  FIXME: RANDINT with inclusive HI would make more sense for matlab
function [R] = randrange(lo,hi,M,N,P);
if nargin == 2
  R = rand();
elseif nargin == 3
  R = rand(M);
elseif nargin == 4
  R = rand(M,N);
else
  R = rand(M,N,P);
end
R = lo + floor(R*(hi-lo));

