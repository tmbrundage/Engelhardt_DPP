% plotRvm  plot rvm input and result
%
%  plotRvm(OPTS,STATS)  will plot the rvm results with the default options
%
%  plotRvm(OPTS,STATS,HOW)  will try to plot according to the options
%  specified in struct HOW. HOW.bare = false will result in plotting a
%  legend and some additional infos.
% 
%  See README.txt for detailed description.
%  copyright 2006 Alexander Schmolck and Richard Everson
function plotRvm(opt, stats, how)
if nargin < 3, how = struct(); end
% $$$ bar(log10(stats.aalpha));
% $$$ title('log_{10} \alpha_m');
PPhi = mygetfield(opt, 'PPhi', @() makeKernel(opt.kernel, opt.N, struct('r', opt.r)));
bare = mygetfield(how, 'bare', mygetfield(opt, 'bare', true));
dataFirst = mygetfield(how, 'dataFirst', mygetfield(opt, 'dataFirst', false));
if ~bare
  figure();
end
N = length(opt.tt);
disp plotting
%x = [1:N]/N;
x = [1:N];
clf();
hold on;
leg = {};
if N>256,
  MS = 3;
  LW = 1;
else
  MS = 10;
  LW = 2;
end
if isfield(opt, 'yy')
  %assert(opt.sigma == std(opt.tt - opt.yy));
  mseStr = sprintf(' mse: %7.6f', mean((opt.yy - stats.yy_hat).^2));
  plot(x, opt.yy, 'k--', 'LineWidth', LW);
  leg{end+1} = '{\bf{}y} true signal';  
  sigmaStr=sprintf(' \\sigma_{rat}:%4.3f',sqrt(stats.sigma_sq)/opt.sigma);
else
  mseStr = ''; sigmaStr = '';
end
if dataFirst
  plot(x, stats.yy_hat, 'm', 'LineWidth', LW+1);
  leg{end+1} = '{\bf{{}_{\fontsize{14}{y}}^{\^}}} posterior estimate';
  plot(x, opt.tt, '.', 'MarkerSize', MS, 'LineWidth', LW);
  leg{end+1} = '{\bf{}t} targets';
else
  plot(x, opt.tt, '.', 'MarkerSize', MS, 'LineWidth', LW);
  leg{end+1} = '{\bf{}t} targets';
  plot(x, stats.yy_hat, 'm', 'LineWidth', LW+1);
  leg{end+1} = '{\bf{{}_{\fontsize{14}{y}}^{\^}}} posterior estimate';
end

%leg{end+1} = '{\bf{}y}_{est}';
%leg{end+1} = '{\bf{t}_hat}';
if ~bare
  % FIXME should think of something better; esp. for else part
  if ~any(strcmp(mygetfield(opt, 'kernel', ''), {'symmlet', 'haar'})) && length(stats)==length(x)% XXX
    plot(x(stats.sel), opt.tt(stats.sel), 'o', ...
         'MarkerSize', MS+2, 'LineWidth', LW+1);
    leg{end+1} = '''Support vectors''';
  else
    centers = map(@(i) argmax(abs(PPhi(:,i))),  find(stats.sel));
    plot(x(centers), stats.yy_hat(centers), 'x', ...
         'MarkerSize', MS+2, 'LineWidth', LW+1);
    ax = axis; height = ax(4) - ax(3);
% $$$   ts = 1./stats.aalpha(stats.sel);
% $$$ % $$$   ts=log(stats.ttheta(stats.sel));
% $$$   errplot(x(centers), stats.yy_hat(centers),...
% $$$           0.3.*height.*ts./max(ts));
    leg{end+1} = 'Component centers';
  end;
  l=legend(leg, 'Location', 'BestOutside'); set(l, 'FontSize', 14);
  [L,P,B,J]=unarray(map(@(n) aref(mygetfield(stats, n),-1),{ 'L_mathcal', 'Post', ...
                    'BIC', 'PSJ'}));

  title(sprintf('N=%d S=%d L=%8g P=%8g B=%8g J=%8g%s%s', N, sum(stats.sel), ...
                L,P,B,J, mseStr, sigmaStr),'FontSize',10);
else
  hold off;
  axis off;
end
hold off

function errplot(x,y,err, style)
if nargin == 3, style = 'b'; end
if isvector(err) == 1, err = [err(:)./2,err(:)./2]; end
lo = y - err(:,1);
hi = y + err(:,2);
bak = ishold;
hold on
for i=1:length(x)
  plot([x(i), x(i)], [lo(i), hi(i)], style);
end
if ~bak, hold off; end

function PostPlot(opt,stats)
leg = {'', '\sigma^2_{est}'};
plot(stats.grad__sigma_sqs(1:end-(1+isfield(opt,'sigma'))), stats.RIC(1:end-(1+isfield(opt,'sigma'))), 'bx', ...
     stats.grad__sigma_sqs(end-isfield(opt,'sigma')), stats.RIC(end-isfield(opt,'sigma')), 'gx')
if isfield(opt, 'sigma'),
  withHold(@() plot(stats.grad__sigma_sqs(end), stats.RIC(end), 'ro'));
  leg{end+1} = '\sigma^2_{real}';
end
%legend(leg)
title('L')
xlabel('sigma^2')
drawnow;
