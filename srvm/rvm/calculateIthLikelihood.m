function l_mathcal=calculateIthLikelihood(alpha, q, s)
l_mathcal = 0.*alpha;
l_mathcal(alpha==inf) = 0;
ninf=0.5.*(log(alpha)-log(alpha+s)+q.^2./(alpha+s));
l_mathcal(alpha~=inf) = ninf(alpha~=inf);

