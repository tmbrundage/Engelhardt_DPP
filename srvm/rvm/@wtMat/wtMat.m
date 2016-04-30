% Make (forward/inverse) wavelet transform appear like matrix multiplication;
% since this is conceptually correct (and often convenient), but much less
% efficient in practice. So this class gives e.g. a handy standin for a kernel
% matrix.
% 
% e.g. >> X = wtMat(N, 'Symmlet', 128); 
%      >> y = sin(linspace(1,6,128));
%      >> plot(X(:[1:10])*(X(:,[1:10])'*y));
%      >> ;; project unto a subset of the symmlet basis vectors and restore
%      >> plot(X(:,[1,2,3,5,6])*(X(:,[1,2,3,5,6])'*y));
%      >> ;; again, using 4 instead of 5 -- with much better result
%      >> plot(X(:,[1,2,3,4,6])*(X(:,[1,2,3,4,6])'*y));
%      >> ;; using all components there is nearly no loss
%      >> max(abs(X*(X'*y)-y))
%      
%      ans =
%      
%         5.0482e-13
%        

function X=wtMat(N,type,No,sel,L)
assert(2^ceil(log2(N))==N, 'N must be power of 2');
if nargin<5,L=1;end;
if nargin<4,sel=':'; end
if nargin<3,No=8; end
if nargin<2,type = 'Symmlet';end;
X.filt=MakeONFilter(type,No);
X.sel = sel;
X.N=N;
X.transposed = false;
X.L=L;
X=class(X,'wtMat');