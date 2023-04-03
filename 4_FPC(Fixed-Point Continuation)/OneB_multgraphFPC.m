clc
% Step 1: Generate X = U*V.'
mn = 10;
rk = 2;
U = randn(mn, rk);
V = randn(mn, rk);
X = U*V.';
[Xx_size, Xy_size]=size(X);
A=1:numel(X);
maxiter = 100;
tol = 1e-5;
RMSE_all = zeros(5, maxiter);

% Specify five missing percentages
missingpers = [10 20 30 40 50];
tau = 1e-6;
rho = 0.9;
mu = 1;

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
    
    % Fixed-Point Continuation Algorithm
    lambda = 1/sqrt(max(Xx_size, Xy_size));
    L = zeros(size(X));
    S = zeros(size(X));
    iter = 0;
    converged = false;
    Y = X_Omega;
    while ~converged && iter < maxiter
        iter = iter + 1;
        % update L using the fixed-point iteration with continuation
        L = fixed_point_iteration(Y - S, lambda);
        % update S using the soft-thresholding operator
        S = soft_threshold(Y - L, 1);
        % update Y by subtracting the difference between X and L + S from Y
        Y = Y + X_Omega - L - S;
        % update the continuation parameter
        mu = rho*mu;
        % update the regularization parameter
        lambda = max(lambda/mu, tau);
        % calculate RMSE
        RMSE_all(i, iter) = sqrt(sum(sum((L - X).^2.*Omega))/num_remove);
        % check for convergence
        if iter > 1 && abs(RMSE_all(i, iter)) < tol
            converged = true;
        end
    end
    
    % Find minimum iteration to achieve RMSE equal to tolerance
    miniter = find(RMSE_all(i,:) <= tol, 1);
    if isempty(miniter)
        disp(['Could not find minimum iteration for ', num2str(missingper), '%. RMSE values may increase over time.'])
    else
        disp(['Matrix ', num2str(mn), 'x', num2str(mn), ' with rank ', num2str(rk), ', missing ', num2str(missingper), '%'])
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
title(['FPC Algorithm with different missing percentages for ', num2str(mn), 'x', num2str(mn), ' matrix']);

% Functions used in the Fixed-Point Continuation Algorithm
function L = fixed_point_iteration(X, lambda)
    [U, S, V] = svd(X, 'econ');
    S = soft_threshold(S, lambda);
    L = U*S*V';
end

function S = soft_threshold(X, lambda)
    % perform soft-thresholding on the matrix X with threshold lambda
    % the result is the matrix S
    S = sign(X).*max(abs(X) - lambda, 0);
end