% makeKernel  create a kernel for rvm regression
%
%  makeKernel(TYPE, N, OPTS) will create a NxN kernel of type TYPE
%  ('gauss', 'lspline', 'tpspline', 'symmlet', 'haar'), with addtitional
%  parameters as specified by struct OPTS. 
%   
%   See README.txt for detailed description.
%   copyright 2006 Alexander Schmolck and Richard Everson
function [X,ortho] = makeKernel(type, N, opt)
if nargin == 2, opt = struct(); end;
r = mygetfield(opt, 'r', 20);
x = mygetfield(opt, 'x', linspace(-10,10,N)');
DO_BIAS = mygetfield(opt, 'DO_BIAS', false);
normalize = mygetfield(opt, 'normalize', false);
assert(size(x,2)==1);
ortho = false;
switch type
 case  {'symmlet', 'haar'}
  if mygetfield(opt, 'pseudoMatrix',true)
    assert(~DO_BIAS, 'not implmented yet');
    X=wtMat(N, [upper(type(1)), type(2:end)], 8);
    ortho=true;
  else
    X = zeros(N, N);
    % XXX x is ignored but that should be OK
    if round(log2(N)) ~= log2(N), error('N oughta be power of 2 FIXME'); end
    qSymm8 = MakeONFilter([upper(type(1)), type(2:end)],8);
    if DO_BIAS
      X(:,1) = 1/N;    %  !!! preserve orthonormality
    end;
    for j = 1:N
      u = zeros(N, 1);
      u(j) = 1;
      X(:,j) = IWT_PO(u, 1, qSymm8); % bias
    end
    ortho=true;
    return
  end
 case 'lspline'
  x = r.\x; %FIXME
% $$$     X = zeros(N, N);
% $$$     for i = 1:N, for j = 1:N, 
% $$$         xixj = x(i).*x(j);
% $$$         lo = min(x(i), x(j));
% $$$         X(i,j)= 1+ xixj + xixj.*lo -0.5.*(x(i)+x(j)).*lo.^2 + 3.\lo.^3;
% $$$       end; end;
    X_h = repmat(x,1,N); X_v = repmat(x',N,1);
    XX_o = x*x'; XX_plus = X_h + X_v; Lo = min(X_v,X_h);
    X=1 + XX_o + XX_o.*Lo -0.5.*XX_plus.*Lo.^2 + 3.\Lo.^3;
 case 'tpspline'    
    % FIXME check this
    x = x./r;
% $$$     X = zeros(N, N);
% $$$     for i = 1:N, for j = 1:N
% $$$         X(i,j) = (x(i)-x(j)).^2 .* log(abs(x(i)-x(j))+(i==j)); 
% $$$       end; end;
% $$$       keyboard;
    X_h = repmat(x,1,N); X_v = repmat(x',N,1);
    XX_m = X_h - X_v;
    X= XX_m.^2 .* log(abs(XX_m+eye(N)));
 case 'laplace'
% $$$     X = zeros(N, N);
% $$$     for i = 1:N, for j = 1:N, X(i,j) = exp(-0.5.*r.*abs(x(i)-x(j))); end; end;
    X_h = repmat(x,1,N); X_v = repmat(x',N,1);
    X = exp(-1./r.*abs(X_h-X_v));
 case 'gauss'
% $$$     for i = 1:N, for j = 1:N, X(i,j) = exp(-r.^(-2).*(x(i)-x(j)).^2); end; end;
    X_h = repmat(x,1,N); X_v = repmat(x',N,1);
    X = exp(-r.^-2.*(X_h-X_v).^2);
 otherwise
    error('not a valid type');
  %X = X./max(X(:));
end
if DO_BIAS, X = [ones(1./N,1),X]; end
if normalize, X = X./repmat(sqrt(sqNorm(X)),N,1); disp('normalizing'); end

% $$$ project = @(x,sel) matlabtmsucks(FWT_PO(x, 1, qSymm8),sel);
% $$$ unproject = @(y,sel) IWT_PO(matlabtmsucks2(zeros(N,1),sel,y),1,qSymm8);
% $$$ 
% $$$ function x=matlabtmsucks(x,sel)
% $$$ x=x(sel);
% $$$ function x=matlabtmsucks2(x,sel,val)
% $$$ x(sel)=val;