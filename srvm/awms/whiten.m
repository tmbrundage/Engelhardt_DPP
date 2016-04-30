% FIXME this is crappy code
function XX_tilde = whiten(XX)
[UU,LLamda]=svd(cov(XX));
XX_tilde = (diag(diag(LLamda).^(-0.5))*UU'*demeaned(XX,2))';
