% rvm  sRVM implementation
%   
%   rvm(rvmopts(struct('tt', tt, 'priorType', 'RIC', 'kernel', 'symmlet')))
%   will do sRVM regression with symmlet kernel and RIC prior on a supplied
%   set of targets tt. 
%   
%   See README.txt for detailed description.
%   copyright 2006 Alexander Schmolck and Richard Everson

%%%_. rvm
function stats = rvm(opt)
revision = '$Revision: 1.27 $';
global step
%%%_ : INITIALIZATIONS
rand('state', opt.seeds(2));
randn('state', opt.seeds(2));
tt = opt.tt(:);
N = length(tt);
opt.chatty = mygetfield(opt, 'chatty', true);
chatty = opt.chatty; %FIXME
stats.date= datestr(now, 'yyyy-mm-ddTHH:MM:SS'); % XXX not GMT, but never mind
opt.date = stats.date;
time_0 = clock();
%%%_  , remove the mean of the signal
%%% XXX really this should be estimated along with all the other coefficients.
if opt.zeroMean,
  av = mean(tt);
  tt = tt - av;
end
if isfield(opt,'yy'),
  if opt.zeroMean,
    yy = opt.yy - av;
  else
    yy = opt.yy;
  end
end
%%%_  , initialize sigma_sq
if opt.reest__sigma
  if opt.reest__sigma == 1 && chatty
    warning('reest__sigma=1 will likely give very bad results; better set reest__sigma=10!!!')
  end
  ssigma = zeros(0,0); % store estimates here
  if ~isfield(opt, 'sigma_0')
    sigma_sq = (std(tt)/100).^2; % just some sensible default
    if chatty
      fprintf('initial guess for sigma_sq: %7.4f (sqrt: %7.4f)', sigma_sq, ...
              sqrt(sigma_sq));
    end
  else
    if imag(opt.sigma_0) % initialize as multiple of known real std
      assert(~real(opt.sigma_0));
      sigma_sq = (opt.sigma .* imag(opt.sigma_0)).^2;
    else
      sigma_sq = opt.sigma_0.^2;
    end
    if chatty, 
      fprintf('sigma_sq initialized from opt.sigma_0 to: %7.4f (sqrt: %7.4f)\n', ....
                       sigma_sq, sqrt(sigma_sq));
    end
  end
  sigma_sq_prime = pastVals(4); %FIXME check what fiddling with this yields
else
  if imag(opt.sigma_0)
    assert(~real(opt.sigma_0));
    sigma_sq = (opt.sigma .* imag(opt.sigma_0)).^2;
  else
    sigma_sq = opt.sigma_0.^2;
  end
  if chatty, 
    fprintf('sigma_sq fixed and set to: %7.4f (sqrt: %7.4f)\n', sigma_sq, ...
            sqrt(sigma_sq));
  end
end
ssigma(1) = sqrt(sigma_sq);
%%%_  , initialize various other stuff
% !!! somewhat confusingly ``NxM`` **not** ``MxN`` (following Tipping)
if ~isfield(opt, 'PPhi'),
  [PPhi, ortho] =  makeKernel(opt.kernel, N, opt);
  ortho = mygetfield(opt, 'ortho', ortho);
else
  PPhi = opt.PPhi;
  ortho = mygetfield(opt, 'ortho', false);
end
a = opt.a; b = opt.b; % hyperparameters for prior

M = size(PPhi,2);
sel = logical(zeros(M,1));
sels = zeros(0,M);
reestSteps = zeros(0,1);
aalpha = zeros(M,1) + inf;
rel = false(M,1);
shouldCalculateL_mathcal = mygetfield(opt, 'shouldCalculateL_mathcal',true); %XXX
if ~shouldCalculateL_mathcal
  L_mathcal = NaN; AIC=NaN; RIC=NaN; BIC=NaN; Post=NaN; PSJ=NaN; L_mathcal_stepwise=NaN;
end
l_mathcal = -inf;
post = -inf;
ll_mathcal_stepwise = nan(1,M);

ll_mathcal = nan(1E5,1);
LL_mathcal = nan(1E5,1);
L_mathcal = NaN;
Post = NaN;
Posts = nan(1E5,1);
posts = nan(1E5,1);
mses = nan(1E5,1);
posts__unnormalized = nan(1E5,1);

% precompute frequently needed results;
PPhi_t_tt = PPhi'*tt;
if ortho,
  kk = 'dummy';
else
  kk = sqNorm(PPhi)';
end

%%%_  , compute Least Squares solution, if desired
if mygetfield(opt, 'computeLS', false),
  if ortho
    stats.mmu_LS = PPhi_t_tt;
  else
    stats.mmu_LS = PPhi\tt;
  end
  stats.tt_LS = PPhi*stats.mmu_LS;
end

%%%_  , Pick first component ``i`` and initialize aalpha, sel, rel & PPhi_now

if ~isfield(opt, 'aalpha')
  if ortho
    PPhi_sqNorm = ones(1,M);
  else
    PPhi_sqNorm = sqNorm(PPhi);
  end
  i = argmax((PPhi_t_tt).^2./PPhi_sqNorm');
  % .2.b. initialize `i` with lowest projection
  sel(i) = true;
  % .2.c initialize corresponding alpha
  PPhi_now = PPhi(:,i);
  aalpha(i) = PPhi_sqNorm(i)./( (PPhi(:,i)'*tt).^2./PPhi_sqNorm(i) - sigma_sq );
  assert(aalpha(i) > 0);
else
  if chatty,  disp ('POLISHING PREEXISTING aalpha'); end;
  aalpha = opt.aalpha;
  sel = aalpha ~= inf;
  rel = sel;     %XXX
  PPhi_now = PPhi(:,sel);
end
kassert(length(aalpha)==length(sel) && length(aalpha)==M)  

%%%_  , Initialize cache (depends on first ``i``)
if ortho
  PPhi_now_t_PPhi = 'dummy';
  K_cache = 'dummy';
else
  K_cache = nan(size(PPhi));
  K_cache(sel,:) = PPhi(:,sel)'*PPhi;
  PPhi_now_t_PPhi = K_cache(sel,:);
end

%%%_  , Explicitly compute Sigma etc. for initialization

[SSigma, mmu, pp, uu] =fullUpdate(PPhi_now_t_PPhi, PPhi_t_tt, kk, sigma_sq, aalpha, sel,ortho);
if ortho, SSigma_diag = SSigma; else SSigma_diag = diag(SSigma);end
[qq, ss] = compute_qq__ss(aalpha, pp, uu, opt);
aalpha_prime = aalpha;

%%%_  , initial L_mathcal
LL_mathcal(1) =  calculateL_mathcal(aalpha(sel), sigma_sq, PPhi_now, mmu, tt, ortho);
Posts(1)      =  LL_mathcal(1) + calculatePrior(aalpha, sigma_sq, N, opt.priorType,false,a,b);
% $$$ Posts_r(1)      =  LL_mathcal(1) + calculatePrior(aalpha, sigma_sq, N, opt.priorType,false,a,b,SSigma,PPhi_now,ortho);
%%%_ : MAIN LOOP
reestimating = false;
toReestimate = [];
shouldStop = false;

realStep = 1; % i.e. a step for which some alpha_i actually changed; the first
              % step already happened
for step = 1:opt.maxSteps
  if ~reestimating
    i = randrange(1,M+1);
  else
    i = toReestimate(reestimating);
    reestimating = reestimating + 1;
    if chatty>2, fprintf('<Reestimating: %d>\n', i); end;
  end
%%%_  , compute "relevance"
  rel_old = rel;
  [rel, alpha_star_i, post__unnormalized] = computeRelevance(N, i, qq, ss, sigma_sq, rel, opt);
  assert((alpha_star_i <= 1E10) == isfinite(alpha_star_i));
  % .6.,.7.,.8. 
  if rel(i)
    assert(alpha_star_i ~= inf);    
    %kassert(all(aalpha >= 0), 'negative alpha');
    if (aalpha(i) < inf)
      %assert(sel(i));
      lastAction = 'reest';
    else
      %assert(~(sel(i)));
      sel(i) = true;
      PPhi_now = PPhi(:,sel);
      lastAction = 'add';
    end
  else
    if aalpha(i) < inf
      %assert(sel(i));
      kassert(sum(sel) >= 1, 'oops, too much irrelevance');
      assert(alpha_star_i == inf);      
      sel(i) = false;
      PPhi_now = PPhi(:,sel);
      lastAction = 'delete';
    else
      %assert(~sel(i));      
      lastAction = 'noop';
    end
  end
  if ~strcmp(lastAction, 'noop') % && ~reestimating % FIXME
    sel_old = aalpha~=inf;
    aalpha_old = aalpha;
    aalpha(i) = alpha_star_i;
    realStep = realStep+1;
    l_mathcal = calculateIthLikelihood(alpha_star_i,qq(i),ss(i));
    post = l_mathcal + calculatePrior(alpha_star_i, sigma_sq, N, opt.priorType,true,a,b);
% $$$     post_r = l_mathcal + calculatePrior(alpha_star_i, sigma_sq, N, opt.priorType,true,a,b,SSigma,PPhi_now,ortho);
    ll_mathcal(realStep-1) = l_mathcal;
    posts(realStep-1) = post;
% $$$     posts_r(realStep-1) = post_r;    
%%%_  , update cache if required
    if ~ortho 
      switch lastAction,
       case 'add'
        K_cache(i,:) = PPhi(:,i)'*PPhi;
        PPhi_now_t_PPhi = K_cache(sel,:);
       case 'delete'
        PPhi_now_t_PPhi = K_cache(sel,:);
      end
    end
%%%_  , .10. recompute [Sigma, mu, u, p]
    SSigma_old = SSigma; SSigma_diag_old = SSigma_diag; mmu_old = mmu; pp_old = pp; uu_old = uu; sigma_sq_old=sigma_sq;
    [SSigma, mmu, pp, uu] = fullUpdate(PPhi_now_t_PPhi, PPhi_t_tt, kk, sigma_sq, aalpha, sel, ortho);
    if ortho, SSigma_diag = SSigma; else SSigma_diag = diag(SSigma);end
    qq_old = qq; ss_old = ss;
    [qq, ss] = compute_qq__ss(aalpha, pp, uu, opt);
    L_mathcal = calculateL_mathcal(aalpha(sel), sigma_sq, PPhi_now, mmu, tt, ortho);
    LL_mathcal(realStep) = L_mathcal;
    PostBit = calculatePrior(aalpha, sigma_sq, N, opt.priorType,false,a,b);
% $$$     PostBit_r = calculatePrior(aalpha, sigma_sq, N, opt.priorType,false,a,b,SSigma,PPhi_now,ortho);    
    Post = L_mathcal + PostBit;
% $$$     Post_r = L_mathcal + PostBit_r;    
    Posts(realStep)=Post;
% $$$     Posts_r(realStep)=Post_r;    
    posts(realStep-1)=post;    
% $$$     posts_r(realStep-1)=post_r; % XXX check this is recomputed
    posts__unnormalized(realStep-1) = post__unnormalized;
    assert(all(isfinite([Post])));
    if isfield(opt, 'yy')
      mse = mean((yy - (PPhi_now * mmu)).^2);
      mses(realStep) = mse;
    end
    if step>=10.*N && Posts(realStep) < Posts(realStep-1) && (~opt.reest__sigma || mod(realStep,opt.reest__sigma)~=0),
      if chatty
        fprintf('***Post failure s:%di:%d (%5s)Post rat: %10.6f post_unnor: %10.6f theta:%5.3g old:%5.3g s: %5.3g old: %5.3g\n', ...
                step, i,lastAction, Posts(realStep)./Posts(realStep-1),  ...
                post__unnormalized, ...
                qq(i)^2-ss(i), qq_old(i)^2-ss_old(i), ss(i), ss_old(i));
      end
      %if strcmp(lastAction, 'add'),        keyboard; end;
      %disp('**BAD CHANGE; reconsidering**')
      if true, %false %step>10*N,
        if chatty, fprintf('UNDOING...\n'); end
        [rel, sel, aalpha, SSigma, SSigma_diag, mmu, pp, uu, sigma_sq, qq, ss, Post] = ...
            deal(rel_old, sel_old, aalpha_old, SSigma_old, SSigma_diag_old, mmu_old, pp_old, uu_old, sigma_sq_old, qq_old, ss_old, Posts(realStep-1));
        Posts(realStep)=NaN;
        posts(realStep-1)=NaN;        
        posts__unnormalized(realStep-1) = NaN;
        LL_mathcal(realStep)=NaN;      
        realStep=realStep-1;
        PPhi_now = PPhi(:,sel);      
        if ~ortho 
          PPhi_now_t_PPhi = K_cache(sel,:);
        end
        lastAction = 'noop';
      end
    end
  end
%%%_ , 9. update noise if desired idea: don't reestimate too often, but make
  % sure that convergence only occurs after noise reestimation once all alpha
  % have been recomputed. ``step<N`` is just a hack to prevent initial
  % bloating right at the beginning due to wrong (too low) noise estimate that
  % otherwise wouldn't get corrected during alpha reestimation
  if opt.reest__sigma && ((~strcmp(lastAction, 'noop') && ...
                       (~reestimating || step < N) && (mod(realStep,opt.reest__sigma)==0)) || ...
                      (reestimating && (reestimating == length(toReestimate))))
    epsilon_sq = sqNorm(PPhi_now*mmu - tt);
    ggamma_sum = sum(1-aalpha(sel).*SSigma_diag);
    switch upper(opt.priorType)
     case {'NONE', 'PSJ'}
      sigma_sq = epsilon_sq ./ (N - ggamma_sum);
     otherwise
      switch upper(opt.priorType)
       case 'NIC', c = 0; 
       case 'AIC', c = 1; 
       case 'BIC', c = log(N)/2; 
       case 'RIC', c = log(N); 
       otherwise assert(isscalar(opt.priorType) & isnumeric(opt.priorType));c=opt.priorType;
      end
      l_noise = @(sigma_sq) 0.5.*(N.*sigma_sq - epsilon_sq - sigma_sq.*ggamma_sum) ...
                - c.*sum(aalpha(sel)./(1./sigma_sq + aalpha(sel)).^2) ...
                + (a-1).*sigma_sq - b;
      % log pi(lamba,beta) = -sum(logZ)
      % D(log pi, beta) approx= -c .* sum(aalpha./(beta+aalpha).^2) + (a-1)./beta - b
% $$$         l_noise = @(sigma_sq) sigma_sq .* sum(aalpha(sel).*diag(SSigma)+(a-1)) - epsilon_sq+b - ...
% $$$                   c.*sum(aalpha(sel)./(sigma_sq.^(-2)+aalpha(sel)).^2);
      try
        sigma_sq = fzero(l_noise, sigma_sq);
        if sigma_sq <0,
          warning('fzero sigma_sq came up with something negative; retry');
          sigma_sq = fzero(l_noise, [eps, 1E10]); %HACK FIXME
          assert(sigma_sq > 0);
        end
      catch
        disp('fzero died!!!');
        keyboard;
      end
    end
    sigma_sq_prime = add(sigma_sq_prime, sigma_sq);
    ssigma(end+1) = sqrt(sigma_sq);
    sigma_sq_star = epsilon_sq ./ (N - ggamma_sum);
    kassert(sigma_sq > 0 && sigma_sq_star > 0); % && sigma_sq_prime > 0);
    SSigma_old = SSigma; SSigma_diag_old = SSigma_diag; mmu_old = mmu; pp_old = pp; uu_old = uu; sigma_sq_old=sigma_sq;
    [SSigma, mmu, pp, uu] = fullUpdate(PPhi_now_t_PPhi, PPhi_t_tt, kk, sigma_sq, aalpha, sel, ortho);
    if ortho, SSigma_diag = SSigma; else SSigma_diag = diag(SSigma);end
    qq_old = qq; ss_old = ss;
    [qq, ss] = compute_qq__ss(aalpha, pp, uu, opt);
  else
    %recomputeEverything = false; % FIXME
  end

%%%_  , 11. check for convergence
%%%     this is all a mess...

% $$$   notMuchChanged = (all(abs(log(aalpha(sel)) - log(aalpha_prime(sel))) < 1E-6) && ...
% $$$                     ~any(rel(~sel)) && all(rel(sel)));
  notMuchChanged = (all(abs(log(aalpha(sel)) - log(aalpha_prime(sel))) < 1E-6));
  if reestimating
    %if ~notMuchChanged, reestimating = 0; end
  else
    if notMuchChanged
% $$$       toReestimate = shuf(find(sel));
% $$$       toReestimate = randperm(M);      
      Post_prime = Post;
      toReestimate = [shuf(find(sel));shuf(find(rel & ~sel));randperm(M)'];
      reestimating = 1;
      reestSteps(end+1)=realStep;
    end
  end
  if reestimating && reestimating == length(toReestimate)
    delta_aalpha = abs(log(aalpha(sel)) - log(aalpha_prime(sel)));
    max__delta_aalpha = max(delta_aalpha);
    if chatty,
      fprintf('###DEBUG Post log diff to Post_prime: %f %f', log(abs(Post./ ...
                                                        Post_prime)));
    end
    shouldStop =  max__delta_aalpha < 1E-6 && ~any(rel(~sel)) && all(rel(sel)) && ...
        (~opt.reest__sigma || maxAbsLogDiff(sigma_sq_prime) < 1E-6) && ...
        (abs(log(abs(Post./Post_prime)))<1e-6)&& ...
        step >= mygetfield(opt, 'minSteps', 10.*N+2); %FIXME
    %FIXME Post./Post_prime should presumably be  Posts(realStep)./Posts(realStep-1)
    
    if ~shouldStop
      if chatty
        fprintf('failed to converge aalpha: %d rel: %d sigma_sq %d\n',...
                ~(max__delta_aalpha < 1E-6), ~(~any(rel(~sel)) && all(rel(sel))),...
                ~(~opt.reest__sigma || maxAbsLogDiff(sigma_sq_prime) < 1E-6))
      end
      if chatty      
        if max__delta_aalpha >= 1E-6,
          fprintf('max delta_aalpha: %10.8f\n', max__delta_aalpha);
        else
          if opt.reest__sigma,
            fprintf('max delta_sigma: %10.8f\n', max(log(vals(sigma_sq_prime))));
          end
        end
      end
      reestimating = false;
      toReestimate = [];
    end
  else
    assert(~reestimating || length(toReestimate)>reestimating)
  end
  if chatty && (~strcmp(lastAction, 'noop') && mod(realStep, 10) == 0 || shouldStop || reestimating == 1)
    tmp = [1:M];
    if isfield(opt, 'yy')
      mse = mean((yy - (PPhi_now * mmu)).^2);
      if ~strcmp(lastAction, 'noop')
        mses(realStep) = mse;
      end
      mseStr = sprintf(' mse: %7.6f', mse);
    else
      mseStr = '';
    end
    if opt.reest__sigma
      if ~isfield(opt, 'sigma')
        sigmaStr = sprintf(' sigma_sq: %10.8f', sigma_sq);
      else
        sigmaStr = sprintf(' sigma_rat: %10.8f', sqrt(sigma_sq)./opt.sigma);
      end
    else
      sigmaStr = '';
    end
    fprintf('step: (%6d|%5d) %-5s %s %s Post:%9.5f(+%9.5f) L_mathcal:%9.5f(+%9.5f) sel(S=%d): %-4s\n', ...
            step, realStep, lastAction, sigmaStr, mseStr, Post, post, L_mathcal, l_mathcal, size(tmp(sel),2), num2str(tmp(sel)));
    if chatty > 1,
      fprintf('\nsel aalpha./(ss.^2./(qq.^2-ss)): %s \n', num2str((aalpha(sel)./(ss(sel).^2./(qq(sel).^2-ss(sel))))',2));
    end
    if reestimating == 1, disp('===About to start reestimation===');end
    if opt.plot > 1
      ploty(PPhi_now * mmu, aref('br', 1+(reestimating==1))); drawnow;
    end
  end
  if shouldStop
    if false && any(strcmp(opt.priorType, {'NIC', 'PSJ'})),
      if chatty, disp('++++Switching prior++++'); end
      shouldStop = false;
      reestimating = false;
      toReestimate = [];
      opt.priorType = 'None';
    else
      break;
    end
  else
% $$$     sels(end+1,:) = aalpha; %XXX
    if ~strcmp(lastAction, 'noop')
      aalpha_prime(i) = aalpha(i);
    end
    
% $$$     if ~(strcmp(lastAction, 'noop') || reestimating > 1)
% $$$       aalpha_prime = aalpha;
% $$$     end
    
  end
end

%%%_ : COMPILE RESULTS
if step == opt.maxSteps
  warning(sprintf('Failed to converge in %d steps (%s)', opt.maxSteps, ...
                  mygetfield(opt, 'stem','')))
  stats.converged = false;
else
  stats.converged = true;
end
%%%_  , final L_mathcal
if shouldCalculateL_mathcal,
  if opt.calculateGradientOfL_mathcal
    grad__sigma_sqs = [linspace(sigma_sq./3,sigma_sq.*3,15), sigma_sq];
    if isfield(opt,'sigma'), 
      grad__sigma_sqs = [grad__sigma_sqs, opt.sigma.^2];
    end
  else
    grad__sigma_sqs = sigma_sq;
  end
  i = 0;
  for grad__sigma_sq=grad__sigma_sqs
    i = i+1;
    L_mathcal(i) = calculateL_mathcal(aalpha(sel), sigma_sq, PPhi_now, mmu, tt, ortho);
    Post(i) = L_mathcal(i) + ...
              calculatePrior(aalpha, grad__sigma_sq, N, opt.priorType,false,a,b);
% $$$     Post_r(i) = L_mathcal(i) + ...
% $$$               calculatePrior(aalpha, grad__sigma_sq, N, opt.priorType,false,a,b,SSigma,PPhi_now,ortho);
    
    AIC(i)  = L_mathcal(i) + calculatePrior(aalpha, grad__sigma_sq, N, 'AIC',false,a,b);
    BIC(i)  = L_mathcal(i) + calculatePrior(aalpha, grad__sigma_sq, N, 'BIC',false,a,b);
    RIC(i)  = L_mathcal(i) + calculatePrior(aalpha, grad__sigma_sq, N, 'RIC',false,a,b);
    try
      PSJ(i)  = L_mathcal(i) + calculatePrior(aalpha, grad__sigma_sq, N, 'PSJ',false,a,b);
    catch
      PSJ(i) = nan;
    end
    if chatty, 
      fprintf('L_mathcal: %8.4f Post: %8.4f BIC: %8.4f PSJ: %8.4f\n',...
              L_mathcal(i), Post(i), BIC(i), PSJ(i));
    end
  end
  assert(all(Post(end-isfield(opt,'sigma')>=Post)));
else
  if chatty, disp('not computing L_mathcal as it would take too long'); end
end
%%%_  , store all the important stuff in stats
stats.revision = revision;
if opt.reest__sigma
  stats.ssigma = ssigma;
else
  stats.ssigma = sqrt(sigma_sq);
end
assert(stats.ssigma(end) == sqrt(sigma_sq));
stats.sigma_sq = sigma_sq;
stats.grad__sigma_sqs = grad__sigma_sqs; %FIXME
stats.sel = sel;
stats.sels = sels;
stats.mmu = mmu;
stats.mmu_M = setind(zeros(1,M), stats.sel, mmu)';
yy_hat = PPhi_now * mmu;
if opt.zeroMean,
  yy_hat = yy_hat + av;
end
stats.yy_hat = yy_hat;
stats.aalpha = aalpha;
stats.ttheta = qq.^2 - ss; %XXX DEBUG
stats.L_mathcal = L_mathcal;
stats.LL_mathcal = LL_mathcal(1:realStep);
stats.Posts = Posts(1:realStep);
% $$$ stats.Posts_r = Posts_r(1:realStep);
%stats.L_mathcal_stepwise = L_mathcal_stepwise;
stats.Post = Post;
stats.AIC = AIC;
stats.BIC = BIC;
stats.RIC = RIC;
stats.PSJ = PSJ;
stats.l_mathcal = l_mathcal;
stats.post = post;
stats.posts = posts(1:realStep-1);
% $$$ stats.posts_r = posts_r(1:realStep-1);
stats.posts__unnormalized = posts__unnormalized(1:realStep-1);
stats.ll_mathcal = ll_mathcal(1:realStep-1);
stats.steps = step;
stats.realSteps = realStep;
stats.reestSteps=reestSteps;
stats.runningTime = etime(clock(),time_0);
if isfield(opt,'yy')
  stats.mse = mean((opt.yy-stats.yy_hat).^2);
end
stats.mses = mses(1:realStep);
%%%_  , plot, print and save if desired
if opt.plot
  plotRvm(opt,stats);
end
if chatty,
  if isfield(opt, 'sigma'), fprintf('sigma_rat: %f\n', sqrt(stats.sigma_sq) / opt.sigma); end
  opt
end
if opt.save
  save([opt.stem, '.mat'], 'stats', 'opt', '-v6');
end


%%%_. fullUpdate
function [SSigma, mmu, pp, uu] = ...
    fullUpdate(PPhi_now_t_PPhi, PPhi_t_tt, kk, sigma_sq, aalpha, sel, ortho)
beta=1./sigma_sq;
if ortho
  SSigma = 1./(aalpha(sel) + beta);
  mmu = beta.*SSigma.*PPhi_t_tt(sel,:);
  uu = beta.*ones(length(aalpha), 1);
  uu(sel) = beta - beta.*beta.*SSigma;
  pp = beta.*PPhi_t_tt;
  pp(sel) = beta.*PPhi_t_tt(sel,:).*(1-beta.*SSigma);
else
  SSigma_inv = diag(aalpha(sel)) + beta .* PPhi_now_t_PPhi(:,sel);
  SSigma = pinv(SSigma_inv);
  if any(isnan(SSigma(:))),
    % HACK matlab BUG workaround FIXME
    SSigma_2 = pinv(diag(aalpha(sel)) + beta .* PPhi_now_t_PPhi(:,sel));    
    if ~any(isnan(SSigma_2(:))),
      disp('DIE MWRKS, DIE!!!')
      SSigma = SSigma_2;
    end
  end
  kassertOk(SSigma(:));
  mmu = beta.*(SSigma_inv\(PPhi_t_tt(sel,:))); % AWMS new  
  PPhi_t_PPhi_now_SSigma = PPhi_now_t_PPhi' *SSigma;
  % Tipping writes: $S_m = beta*`PPhi_m.PPhi ~ beta^2*`PPhi_m.PPhi.SSigma.`PPhi.PPhi_m$:[24]
  uu = beta.*kk - beta.*beta.*sum(PPhi_t_PPhi_now_SSigma .* PPhi_now_t_PPhi',2); % FIXME stupid '
  % Tipping writes: $Q_m = beta PPhi_m' tt - beta.^2 Phi_m' PPhi SSigma PPhi' tt$:[25]
  pp = beta.*PPhi_t_tt - beta.*beta*PPhi_t_PPhi_now_SSigma * PPhi_t_tt(sel,:);
end
kassertOk(uu);
kassertOk(pp);

%%%_. compute_qq__ss

function [qq, ss] = compute_qq__ss(aalpha, pp, uu, opt);
qq = zeros(length(aalpha),1);
ss = zeros(length(aalpha),1);
bad = aalpha==inf;
notBad = ~bad;
ss(bad) = uu(bad);
qq(bad) = pp(bad);
tmp=(aalpha(notBad) - uu(notBad));
ss(notBad) = aalpha(notBad).*uu(notBad)./tmp;
qq(notBad) = aalpha(notBad).*pp(notBad)./tmp;
% $$$ assert(all(ss>0));
bad__ss = ss<=0;
if any(bad__ss), 
  if opt.chatty
    fprintf('!!!!!BAD ss!!!!!\n');
  end
  % !!! Numerical HACK -- turn off dodgy components
  % FIXME: note that this still needs manual intervention for PSJ
  ss(bad__ss) = inf;
end
kassertOk(qq);
kassertOk(ss);
%kassert(all(qq>0));

%%%_. computeRelevance

function [rel, alpha_star_i, post__unnormalized] =  computeRelevance(N, i, qq, ss, sigma_sq, oldRel, opt)
global step;
L = 1E-10; %FIXME HACK
H = 1E+10;
priorType = opt.priorType;
a = opt.a; b = opt.b;
post__unnormalized = 0;
chatty = mygetfield(opt, 'chatty', true);
switch upper(priorType)
 case 'NONE'
  ttheta = qq.^2 - ss;
  rel =  ttheta > 0;
  %rel =  ttheta > 1e-4;   %XXX
  if rel(i),
    s=ss(i); q=qq(i);
    alpha_star_i = s.^2 ./ ttheta(i);
    post__unnormalized=0.5.*(log(alpha_star_i)-log(alpha_star_i+s)+q.^2./(alpha_star_i+s));
  else
    alpha_star_i = inf;
  end  
  return
 case 'PSJ'
  aalpha_star = zeros(length(ss),1);
  DDelta = ss.^2 -6.*ss.*qq.^2 +qq.^4;
  bad = DDelta < 0 | isnan(DDelta); % XXX from the inf ss
  DDelta_sqrt = sqrt(DDelta(~bad));
  aalpha_star(bad) = inf;
  bb = qq(~bad).^2 - 3.*ss(~bad);
  plus =  (bb + DDelta_sqrt); % FIXME DEBUG
  less =  (bb - DDelta_sqrt);

  foo = zeros(sum(~bad),1);
  foo(less>0) = less(less>0);
  foo(less<=0) = plus(less<=0);
  aalpha_star(~bad) = 0.5.*foo;
  %kassert(all(less <= 0 || plus <= 0));
% $$$   alpha_star_i = 0.5.*max(less, plus);
  alsoBad = aalpha_star < 0;
  aalpha_star(alsoBad) = inf;
  rel = ~(bad | alsoBad);
  assert(all(rel==(aalpha_star ~= inf)));
  assert(all(~rel==(aalpha_star == inf)));
  if aalpha_star(i) < L || aalpha_star(i) >H
    aalpha_star(i) = inf;
    rel(i) = false;
  end
  alpha_star_i = aalpha_star(i);
  if ~isreal(alpha_star_i) || alpha_star_i<=0 || (alpha_star_i~=inf) ~= rel(i)
    keyboard;
    error 'bad alpha_star_i in PSJ'; 
  end
  post__unnormalized = 'FIXME';
  return  
 case 'NIC' % None -- the hard way
  c = 0;
 case 'AIC'
  c = 1;
 case 'BIC'
  c = log(N)./2;
 case 'RIC'
  c = log(N);
 otherwise
  assert(isscalar(priorType) & isnumeric(priorType));
  c=priorType;
end
q=qq(i); s=ss(i);
theta = q.^2 - s;
if theta > 0,
  alpha_None = s.^2 ./ theta;
  beta = 1./sigma_sq;  

  B_0 = s.^2*beta.^2;
  B_1 = s.*beta.^2 + 2.*beta.*s.^2 - beta.^2.*q.^2 + 2.*s.^2 .* c.*beta;
  B_2 = 2.*s.*beta + s.^2 - 2.*beta.*q.^2 + 4.*s.*beta.*c;
  B_3 = s - q.^2 + 2.*c.*beta;

  coeffs = [B_3, B_2, B_1, B_0];
  r = roots(coeffs);

  assert(sum(~imag(r)) ~= 2);  %XXX
  myroots = r(r>0 & ~imag(r));
  if length(myroots),
    alpha_star_i_r = min(myroots);
    assert(isreal(alpha_star_i_r));
  else
    alpha_star_i_r = inf;
  end
  alpha_star_i = alpha_star_i_r;
  if alpha_star_i > H
    if chatty>3,
      display('intolerable alpha');    
    end
    alpha_star_i = inf;
  elseif alpha_star_i < L
    if chatty,
      display('XXX too good to be true alpha XXX');
    end
    alpha_star_i = L; %XXX changed that
  else
    post__unnormalized = 0.5.*(log(alpha_star_i)-log(alpha_star_i+s)+q.^2./(alpha_star_i+s)) + -c./(1+sigma_sq.*alpha_star_i);    
  end
  
else
  alpha_star_i = inf;
end

rel = oldRel;
rel(i) = alpha_star_i ~= inf;
if rel(i), assert(alpha_star_i > 0); end

%%%_ : HELPER FUNCTIONS

function assertOk(X)
assert(isreal(X) && ~any(isnan(X)), 'this is unreal man (or nan)');

function kassertOk(X)
kassert(isreal(X) && ~any(isnan(X)), 'this is unreal man (or nan)');

function L_mathcal = calculateL_mathcal(aalpha_now, sigma_sq, PPhi_now, mmu, tt, ortho)
N=length(tt);
if ortho
  lndet__SSigma_inv = sum(log(aalpha_now+1./sigma_sq));
  SSigma_inv = diag(aalpha_now+1./sigma_sq); % XXX
else
  SSigma_inv=diag(aalpha_now)+1./sigma_sq.*PPhi_now'*PPhi_now;
  lndet__SSigma_inv=log(det(SSigma_inv));
end
aa = sigma_sq.\(PPhi_now'*tt);
bb = SSigma_inv\aa;
msim2 = aa'*bb;
L_mathcal= -0.5.*(N.*log(2.*pi) + N.*log(sigma_sq) - sum(log(aalpha_now)) + lndet__SSigma_inv + (1./sigma_sq.*(tt'*tt) - msim2));
kassert(isfinite(L_mathcal));
