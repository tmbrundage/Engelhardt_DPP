% rvmopts  create option structure for rvm
%
%  simplest useful example: rvmopts('tt', some_regression_targets)
%
%  scenarios: - neither yy nor tt => create both from opt.sig 
%             - both
%             - just yy
%             - just tt
%   
%   See README.txt for detailed description.
%   copyright 2006 Alexander Schmolck and Richard Everson

function opt = rvmopts(opt)
DEFAULT_N = 512;
DEFAULT_SIG = 'Doppler';
DEFAULT_SNR = 5.0;
DEFAULT_PRIOR_TYPE = 'BIC';
DEFAULT_KERNEL = 'symmlet';
DEFAULT_R = 3.0;
DEFAULT_REEST_SIGMA = 10;

DEFAULT_DUB = 'test-';
DEFAULT_MAX_STEPS = 1E+6;

% set hyperparameters a,b to somethign uninformative
if ~isfield(opt, 'calculateGradientOfL_mathcal'),
  opt.calculateGradientOfL_mathcal = false;
end
if ~isfield(opt, 'maxSteps'), opt.maxSteps = DEFAULT_MAX_STEPS; end
if ~isfield(opt, 'a'), opt.a = 1E-4; end
if ~isfield(opt, 'b'), opt.b = 1E-4; end 
if ~isfield(opt, 'zeroMean'), opt.zeroMean = true; end %FIXME
if ~isfield(opt, 'seeds'), opt.seeds = randrange([1,2^20],2,1); end
if ~isfield(opt, 'DUB'), opt.DUB = DEFAULT_DUB; end
assert(~(isfield(opt, 'sigma') & isfield(opt, 'SNR')), ...
       'Both sigma and SNR specified (did you mean sigma_0?)');
if ~(isfield(opt, 'yy') || isfield(opt, 'tt'))
  if ~isfield(opt, 'N'), if isfield(opt, 'PPhi'), opt.N=size(opt.PPhi,1); else opt.N = DEFAULT_N; end; end
  if ~isfield(opt, 'sig'), opt.sig = DEFAULT_SIG; end
  if strcmp(opt.sig, 'Sinc')
    x = linspace(-10,10,opt.N); opt.yy = sinc(x./pi)';
  else
    x = linspace(0,1,opt.N);
    opt.yy = MakeSignal(opt.sig, opt.N)';
  end
  if isfield(opt, 'scale') % XXX DEBUG HACK
    opt.yy = opt.yy .* opt.scale;
  end
  s = std(opt.yy);
  if ~isfield(opt, 'SNR'), 
    if ~isfield(opt, 'sigma'), 
      opt.SNR = DEFAULT_SNR;
    else 
      opt.SNR = s./opt.sigma;
    end
  end
  rand('state', opt.seeds(1));
  randn('state', opt.seeds(1));
  if ~isfield(opt, 'sigma'), 
    opt.sigma = s./opt.SNR; 
  end
  opt.tt = opt.yy + std1(randn(opt.N, 1)).*opt.sigma;
elseif isfield(opt, 'yy') && isfield(opt, 'tt')
  opt.N = length(opt.yy);
  s = std(opt.yy);
  if ~isfield(opt, 'sigma'), opt.sigma = std(opt.yy - opt.tt); end
  if ~isfield(opt, 'SNR'), opt.SNR = std(opt.yy)./opt.sigma; end
elseif isfield(opt, 'yy')
  opt.N = length(opt.yy);
  if ~isfield(opt, 'SNR'), opt.SNR = 2; end  
  rand('state', opt.seeds(1));
  randn('state', opt.seeds(1));
  s = std(opt.yy);
  opt.sigma = s./opt.SNR;
  noise = std1(randn(opt.N, 1));
  opt.tt = opt.yy + noise.*opt.sigma;
elseif isfield(opt, 'tt')
  opt.N = length(opt.tt);
  opt.SNR = nan;
else
  error('You just found a bug -- well done!');
end
  
if ~isfield(opt, 'priorType'), opt.priorType = DEFAULT_PRIOR_TYPE; end
if ~isfield(opt, 'kernel') && ~isfield(opt, 'PPhi'), opt.kernel = DEFAULT_KERNEL; end
if ~isfield(opt, 'r') && ~isfield(opt, 'PPhi'), opt.r = DEFAULT_R; end
if ~isfield(opt, 'plot'), opt.plot = true; end
if ~isfield(opt, 'reest__sigma'), opt.reest__sigma = DEFAULT_REEST_SIGMA; end
% !!! Complex sigma is treated as factor
if ~isfield(opt, 'sigma') ,  assert(~exist('sigma', 'var'));
elseif ~isreal(opt.sigma),  assert(real(opt.sigma)==0); opt.sigma = sigma .* opt.sigma * -1j; end
if ~isfield(opt, 'stem'), 
  if strcmp(mygetfield(opt,'kernel', 'symmlet'), 'symmlet')
    rStr = '';
  else
    rStr = sprintf('%.2f', opt.r);
  end
  if mygetfield(opt, 'SUFFIX', false),
    suf = opt.SUFFIX;
  else
    suf = '';
  end
  opt.stem = sprintf('%s%s-SNR%.2f=%.3f-%s%s-%s%s', opt.DUB, mygetfield(opt,'sig', ''), ...
                     mygetfield(opt,'SNR',''), mygetfield(opt, 'sigma', ''), mygetfield(opt,'kernel', 'custom'), ...
                     rStr, opt.priorType, suf);
  opt.stem = strrep(opt.stem, '.', '_');
end      
if ~isfield(opt, 'save'), opt.save = false;end

% Plausibility Checks
if isfield(opt, 'yy'), 
  assert(abs(opt.sigma - std(opt.yy)./opt.SNR) <= 1E-15 && ... 
         abs(opt.sigma - std(opt.tt-opt.yy))   <= 1E-15); 
end
