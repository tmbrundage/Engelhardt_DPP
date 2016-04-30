function f=normalDistribution(mu, sigma)
normal = @(mu sigma) = @(x) (2.*pi.*sigma)^-0.5.*exp(-0.5.*(x-mu)/sigma^2)