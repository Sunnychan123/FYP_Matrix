clc;

% Step1: Generate a random low-rank matrix
mn = 10;   % matrix size
rk = 2;     % rank
U = randn(mn,rk);   % random orthonormal matrix
V = randn(mn,rk);   % random orthonormal matrix
X = U*V.';  % low-rank matrix
[Xx_size, Xy_size] = size(X);

% Step2: Add random missing entries
missingper = 10;    % missing percentage
num_remove = floor((100-missingper)/100*numel(X));
A = 1:numel(X);
omega = A(randperm(numel(A),num_remove));
Omega = sparse(omega, ones(num_remove, 1), 1, Xx_size*Xy_size, 1);
Omega = logical(Omega); 

X_Omega = X;
X_Omega(~Omega) = 0;

% Step3: Set algorithm parameters
lambda = 1/sqrt(max(Xx_size, Xy_size)); % regularization parameter
tol = 1e-6; % tolerance for convergence
maxiter = 100; % maximum number of iterations
mu = 2; % penalty parameter for singular values
rho = 1.5; % penalty parameter for residuals

% Step4: Initialize variables
L = zeros(size(X));
S = zeros(size(X));
Y = X_Omega;
RMSE = zeros(maxiter, 1);
Omega = reshape(Omega, size(Y-X));

% Step5: Run DA algorithm
for iter = 1:maxiter
    % Compute L using SVT-FPC algorithm
    [L, converged] = svt_fpc(Y, lambda, mu, tol, maxiter);
    
    % Compute S using the shrinkage operator
    S = X_Omega - L;
    S(Omega) = soft_threshold(S(Omega), lambda/mu);
    
    % Update Y using the ADMM algorithm
    Y = Y + rho*(X_Omega - L - S);
    Y = Y - rho*(Omega./(1/rho + Omega)).*(Y - X);
    
    % Compute the RMSE
    RMSE(iter) = norm(L - X, 'fro')/sqrt(numel(X_Omega));
    
    % Check for convergence
    if iter > 1 && converged
        break;
    end
    
    % Update mu, rho and Y_prev
    mu = mu*rho;
    rho = min(rho*1.1, 1e6);
end


% Step6: Display the results
% Plot the RMSE versus iteration
figure;
plot(1:maxiter, RMSE(1:maxiter), 'LineWidth', 2);
xlabel('Iteration');
ylabel('RMSE');
title(['DA with ' num2str(missingper) '% missing elements']);

disp(['DA algorithm converged after ' num2str(iter) ' iterations']);
disp(['Final RMSE: ' num2str(RMSE(iter))]);
figure;
subplot(2,2,1);
imagesc(X);
title('Original matrix');
subplot(2,2,2);
imagesc(X_Omega);
title('Observed matrix');
subplot(2,2,3);
imagesc(L);
title('Recovered low-rank matrix');
subplot(2,2,4);
imagesc(S);
title('Recovered sparse matrix');

% Step7: Define helper functions
function [L, converged] = svt_fpc(Y, lambda, mu, tol, maxiter)
% Sparse and low-rank matrix completion using SVT-FPC algorithm
% Y: observed matrix with missing entries
% lambda: regularization parameter for low-rank term
% mu: regularization parameter for sparse term
% tol: tolerance for convergence
% maxiter: maximum number of iterations
% L: completed low-rank matrix
% S: completed sparse matrix
% converged: flag indicating whether the algorithm converged

    [m, n] = size(Y);
    omega = Y ~= 0;
    X_Omega = Y;
    X_Omega(~omega) = 0;
    Xx_size = m;
    Xy_size = n;
    err = Inf;
    L = zeros(m, n);
    S = zeros(m, n);
    converged = false;
    for iter = 1:maxiter
        % Compute low-rank and sparse components using SVT-FPC
        Y = X_Omega - S/mu;
        [Uy, Sy, Vy] = rsvd(Y, 2*mu+1);
        L = Uy*soft_threshold(Sy, lambda/mu)*Vy';
        S = soft_threshold(X_Omega - L, mu);
        % Check convergence
        err_prev = err;
        err = norm(S, 'fro')/norm(X_Omega, 'fro');
        if abs(err - err_prev) < tol
            converged = true;
            % Return all output arguments
            return
        end
    end
end

function [U, S, V] = rsvd(A, k)
% Randomized SVD algorithm
% A: input matrix
% k: number of singular values to compute
% U: left singular vectors
% S: singular values
% V: right singular vectors

    [m, n] = size(A);
    Omega = randn(n, round(k));
    Y = A*Omega;
    [Q, ~] = qr(Y, 0);
    B = Q'*A;
    [U_tilde, S, V] = svd(B, 'econ');
    U = Q*U_tilde;
    U = U(:, 1:k);
    S = S(1:k, 1:k);
    V = V(:, 1:k);
end

function x = soft_threshold(x, tau)
% Soft thresholding operator
% x: input vector or matrix
% tau: threshold value
% x: thresholded output
    x = sign(x).*max(abs(x) - tau, 0);
end