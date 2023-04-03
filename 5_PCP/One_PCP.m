clc
% Step1 generate X = U*V.'
mn=10;
rk=2;
U = randn(mn,rk);
V = randn(mn,rk);
X = U*V.';
Xrank=rank(X);
[Xx_size, Xy_size]=size(X);
A=1:numel(X);
rem10per =floor(0.9*numel(X));
maxiter=100;

% specify missing percentage
missingper = 20;

% calculate number of elements to remove
num_remove = floor((100-missingper)/100*numel(X));

%random permutation of integers start from 10%
omega= A(randperm(numel(A),num_remove));
%matrix zero with Xsize
Omega = zeros (Xx_size,Xy_size);
%change value from 0 to 1 in Omega matrix
Omega (omega) = 1;
%Remove matrix X elements, store at X_Omega
X_Omega = X.*Omega;

% Principle Component Pursuit Algorithm
% set the parameter value
lambda = 1/sqrt(max(Xx_size, Xy_size));
tol = 1e-5;
mu = max(Xrank, 1);
gamma = 1.5; % Initialize the continuation factor
mu_min=1e-6;
% initialize
Y = X_Omega;
W = zeros(size(X));
iter = 0;
converged = false;
RMSE=zeros(1,maxiter);

while ~converged && iter < maxiter
    iter = iter + 1;
    % Update the low-rank matrix estimate 
    % Y using the singular value thresholding operator
    Y = SVT(X_Omega - W, mu);
    % Update the sparse matrix estimate W 
    % using the soft-thresholding operator
    W = soft_threshold(X_Omega - Y, lambda/mu);
    % Calculate the RMSE
    RMSE(iter) = norm(X_Omega - W - Y, 'fro')/norm(X_Omega, 'fro'); 
    % Update the continuation parameter mu
    mu = max(mu/gamma, mu_min);
    % check for convergence
    if norm(X_Omega - W, 'fro')/norm(X_Omega, 'fro') < tol
        converged = true;
    end
end


%find min iteration achieve RMSE approximate to 0
dif = abs(RMSE-tol);
miniter = find(dif == min(dif));
if RMSE(miniter) < tol
    disp(['matrix ', num2str(mn), 'X', num2str(mn), ' with rank',num2str(rk)])
    disp(['Minimum iteration: ' num2str(miniter)])
else
    disp(['Could not find minimum iteration for ',num2str(missingper),'%. RMSE values may increase over time.']);
end

%plot graph
x = 1:maxiter;
figure;
plot(x, RMSE);
xlabel('No. of Iteration');
ylabel('RMSE');
title(['PCP with ' num2str(missingper) '% missing elements']);

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