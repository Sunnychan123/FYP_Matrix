clc
% Step 1: Generate X = U*V.'
mn = 1000;
rk = 2;
U = randn(mn, rk);
V = randn(mn, rk)';
X = U*V;
Xrank=rank(X);
[Xx_size, Xy_size]=size(X);
A=1:numel(X);
maxiter = 100;

% Specify five missing percentages
missingpers = [10 20 30 40 50];
tol = 1e-5;
gamma = 1.5; % Initialize the continuation factor
mu_min=1e-6;
converged = false;
RMSE_all = zeros(5, maxiter);

% Remove specified percentages of elements
for i = 1:length(missingpers)
    missingper = missingpers(i);
    % Calculate number of elements to remove
    num_remove = floor((100-missingper)/100*numel(X));
   
    % Random permutation of integers starting from 10%
    omega = A(randperm(numel(A), num_remove));
    % Matrix of zeros with Xsize
    Omega = zeros(Xx_size, Xy_size);
    % Change value from 0 to 1 in Omega matrix
    Omega(omega) = 1;
    % Remove matrix X elements, store in X_Omega
    X_Omega = X.*Omega;
    
    % PCP algorithm
    lambda = 1/sqrt(max(Xx_size, Xy_size));
    Y = X_Omega;
    W = zeros(size(X));
    iter = 0;
    converged = false;
    mu = max(Xrank, 1);
    while ~converged && iter < maxiter
        iter = iter + 1;
        % Update the low-rank matrix estimate Y 
        % using the singular value thresholding operator
        Y = SVT(X_Omega - W, mu);
        % Update the sparse matrix estimate W
        % using the soft-thresholding operator
        W = soft_threshold(X_Omega - Y, lambda/mu);
        % Update the continuation parameter mu
        mu = max(mu/gamma, mu_min);
        % Calculate the RMSE
        RMSE_all(i,iter) = norm(X_Omega - W - Y, 'fro')/norm(X_Omega, 'fro'); 
        % check for convergence
        if RMSE_all(i,iter) < tol
            converged = true;
        end
    end
    % Reset the RMSE value to zero for the next missing percentage
    RMSE_all(i+1:end, :) = 0;
    % Find minimum iteration to achieve RMSE equal to tolerance
    miniter = find(RMSE_all(i,:) <= tol,1);
    if isempty(miniter)
        disp(['Could not find minimum iteration for ', num2str(missingper), ...
            '%. RMSE values may increase over time.'])
    else
        disp(['Matrix ', num2str(mn), 'X', num2str(mn), ' with rank ', ...
            num2str(rk), ', missing ', num2str(missingper), '%'])
        disp(['Minimum iteration: ', num2str(miniter)])
    end
end

% Plot graph
x = 1:maxiter;
figure;
hold on;
for i = 1:size(RMSE_all, 1)
    y = RMSE_all(i,:);
    missingper = missingpers(i);
    plot(x, y, 'DisplayName', [num2str(missingper), '%']);
end
hold off;
legend('show');
xlabel('No. of Iteration');
ylabel('RMSE');
title(['PCP Algorithm with different missing percentages for ', ...
    num2str(mn), 'x', num2str(mn), ' matrix']);

% Define the singular value thresholding operator
function Y = SVT(X, tau)
    [U, S, V] = svd(X, 'econ');
    S_tau = max(S - tau, 0);
    Y = U * S_tau * V';
end

% define the soft-thresholding operator
function W = soft_threshold(Y, lambda)
    W = sign(Y) .* max(abs(Y) - lambda, 0);
end