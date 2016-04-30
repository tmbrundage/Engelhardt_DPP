function [lnPrior] = calculatePrior(aalpha, sigma_sq, N, priorType, single,a,b, SSigma, PPhi_now,ortho)
trueDF = false;
if nargin == 10,
  trueDF = true; 
else
  assert(nargin==7);
end
%FIXME check this is all correct; esp. single
if single % FIXME
  %assert(length(aalpha) == 1); 
else
  aalpha = aalpha(aalpha ~= inf); 
end 
L = 1E-10; %FIXME the cutoff should be synchronized with inf checks for aalpha
H = 1E+10;
beta = 1./sigma_sq;


% $$$ strlg = 'a*log(b)-gammaln(a)+(a-1)*log(x)-b*x';
% $$$ logGamma = inline(strlg, 'x', 'a', 'b');
ln__p_beta = logGamma(beta, a,b);

switch upper(priorType)
 case 'NONE'
  lnPrior = 0;
  return;
 case 'PSJ'
  lnv = -0.5.*log(aalpha);
  %good = (lnv >= log(L)) & (lnv <= log(H));
  good = (lnv >= 0.5*log(L)) & (lnv <= 0.5*log(H)); %FIXME
  lnv(~good) = inf;
  if single
    lnP = -log(log(H./L))    - lnv;
  else
    lnP = -log(log(H./L)).*N - sum(lnv);
  end
  assert(single|all(good));
 otherwise
  switch upper(priorType)
   case 'NIC',c=0;case 'AIC',c=1;case 'BIC',c=0.5*log(N);case 'RIC',c=log(N);
   otherwise assert(isscalar(priorType) & isnumeric(priorType));c=priorType;
  end
  %fprintf('approx DF: %10.5f\n', sum(-c./(1 + sigma_sq.*aalpha)));
  Z = (beta + H).*exp(-c./(1 + sigma_sq.*H)) - (beta + L).*exp(-c./(1 + sigma_sq .* L));
  if c ~= 0,  % !!! need this because expint gives NaN for c==0
    Z = Z + c.*beta.*( expint(c./(1 + sigma_sq.*L)) - expint(c./(1 + sigma_sq.*H)) );
  end
  good = (aalpha >= L) & (aalpha <= H);      
  if single
    lnP =    -log(Z)    +     -c./(1+sigma_sq.*aalpha);
    lnP(~good) = -inf;
  else
    if trueDF && ~ortho
      DF = trace(sigma_sq.\PPhi_now*SSigma*PPhi_now');
      fprintf('approx DF: %10.5f\n', sum(1./(1 + sigma_sq.*aalpha)));
      fprintf('real   DF: %10.5f\n', DF);      
    else
      DF = sum(1./(1+sigma_sq.*aalpha));
    end
    lnP =    -log(Z).*N + -c.*DF;
    %kassert(all(good));
    if ~all(good) %FIXME
      error('Some alpha outside [L,H] -- prior meaningless');
    end
    %lnP = NaN;
  end
% $$$  otherwise
% $$$   error(sprintf('illegal prior "%s"', priorType));
end
lnPrior = lnP + ln__p_beta;
kassert(single && length(aalpha) > 1 || ...
        all(exp([lnPrior, lnP, ln__p_beta])<=1 & exp([lnPrior, lnP, ln__p_beta])>=0)); %XXX
kassert(lnP<0);

function gam = logGamma(x,a,b)
gam = a.*log(b)-gammaln(a)+(a-1).*log(x)-b.*x;
