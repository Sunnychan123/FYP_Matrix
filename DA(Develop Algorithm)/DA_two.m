clc;

% Step1 generate X = U*V.'
mn=1000;
max_r = mn; % maximum rank to test

maxiter=100;

% specify five missing percentages
missingpers = [10 20 30 40 50];

% initialize array for storing final RMSE values
RMSE_vs_rank = zeros(length(missingpers), max_r);

% Loop over the missing percentages and ranks and run the algorithm for each one
for r = 1:max_r
    % Generate a random low-rank matrix
    rk = 2;
    U = randn(mn,rk); % random orthonormal matrix
    V = randn(mn,rk); % random orthonormal matrix
    X = U*V.'; % low-rank matrix
    [Xx_size, Xy_size] = size(X);

    
    for i = 1:length(missingpers)

        % Add random missing entries
        missingper = missingpers(i);
        num_remove = floor((100-missingper)/100*numel(X));
        A = 1:numel(X);
        omega = A(randperm(numel(A),num_remove));
        Omega = sparse(omega, ones(num_remove, 1), 1, Xx_size*Xy_size, 1);
        Omega = logical(Omega); 
    
        X_Omega = X;
        X_Omega(~Omega) = 0;
    
        % Set algorithm parameters
        lambda = 1/sqrt(max(Xx_size, Xy_size)); % regularization parameter
        tol = 1e-6; % tolerance for convergence
        mu = 10; % penalty parameter for singular values
        rho = 1.5; % penalty parameter for residuals
    
        % Initialize variables
        L = zeros(size(X));
        S = zeros(size(X));
        Y = X_Omega;
    
        % Reshape Omega to have the same size as Y and X
        Omega = reshape(Omega, size(X));
    
        % Run DA algorithm
        converged = false;
        for iter = 1:maxiter
            % Compute L using SVT-FPC algorithm
            [L, converged] = svt_fpc(Y, lambda, mu, tol, maxiter);
    
            % Compute S using the shrinkage operator
            S = X_Omega - L;
            S(Omega) = soft_threshold(S(Omega), lambda/mu);
    
            % Update Y using the ADMM algorithm
            Y = Y + rho*(X_Omega - L - S);
            Y = Y - rho*(Omega./(1/rho + Omega)).*(Y - X);
    
            % Compute RMSE and check for convergence
            RMSE = norm(X - L - S, 'fro')/norm(X, 'fro');
            if converged || RMSE < tol
                break;
            end
        end
    
        % Store final RMSE value
        RMSE_vs_rank(i, r) = RMSE;
    
        fprintf('Rank %d, Missing Percentage %d, RMSE %.4f\n', r, missingper, RMSE);
    end
end

% Plot RMSE vs. rank for each missing percentage
figure();
hold on;
for i = 1:length(missingpers)
    plot(1:max_r, RMSE_vs_rank(i, :), 'LineWidth', 2);
end
xlabel('Rank');
ylabel('RMSE');
title(['DA RMSE vs. Rank with Matrix ' num2str(mn) 'x' num2str(mn)]);
legend(cellstr(num2str(missingpers.', 'Missing = %d%%')), 'Location', 'northeastoutside');
grid on;

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