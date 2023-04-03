clc
% generate a low rank matrix
m = 100;    % number of rows
n = 100;    % number of columns
r = 10;     % rank
X = randn(m,r)*randn(r,n);

% add sparse noise
p = 0.1;    % proportion of entries to corrupt
E = sparse(rand(m,n) < p).*randn(m,n);

% add both low rank and sparse components
Y = X + E;

% set algorithm parameters
lambda = 1/sqrt(max(m,n));
tol = 1e-6;
max_iter = 1000;

% initialize variables
U = zeros(m,r);
V = zeros(n,r);
W = zeros(m,n);
iter = 0;
converged = false;

% run the PCP algorithm
while ~converged && iter < max_iter
    iter = iter + 1;
    % update U and V using SVD
    [U,S,V] = svd(Y - W, 'econ');
    U = U(:,1:r);
    S = S(1:r,1:r);
    V = V(:,1:r);
    % update W using soft thresholding
    W = soft_threshold(Y - U*S*V', lambda);
    % check for convergence
    if norm(Y - U*S*V' - W, 'fro')/norm(Y, 'fro') < tol
        converged = true;
    end
end

% display results
fprintf('Original rank: %d\n', rank(X));
fprintf('Recovered rank: %d\n', rank(U*S*V'));
fprintf('RMSE: %f\n', norm(Y - U*S*V' - W, 'fro')/sqrt(numel(Y)));

% define the soft-thresholding operator
function W = soft_threshold(Y, lambda)
    W = sign(Y) .* max(abs(Y) - lambda, 0);
end
