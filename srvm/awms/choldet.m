% CHOLDET  compute deterimant via cholesky decomposition.
function x = choldet(S)
x=prod(diag(chol(S))).^2;