clc
% Step1 generate X = U*V.'
% mn = 100;
% rk = 2;
% U = randn(mn,rk);
% V = randn(mn,rk);
X = [2 1 0; 0 3 1; 1 0 2; 2 1 0; 0 2 1]
Xrank = rank(X);
[Xx_size, Xy_size] = size(X);
A = 1:numel(X);
rem10per = floor(0.9*numel(X));
maxiter = 100;

% specify missing percentage
% missingper = 10;

% calculate number of elements to remove
% num_remove = floor((100-missingper)/100*numel(X));

%random permutation of integers start from 10%
omega=[1 3 4 5 6 7 8 10 11 12 13 15]
%matrix zero with Xsize
Omega = zeros (Xx_size,Xy_size);
%change value from 0 to 1 in Omega matrix
Omega (omega) = 1
%Remove matrix X elements, store at X_Omega
X_Omega = X.*Omega
missingper=20;
% Singular Value Thresholding Algorithm
lambda = 1/sqrt(max(Xx_size, Xy_size));
L = zeros(size(X));
S = zeros(size(X));
Y = X_Omega;
%tolerance = 1e-5
tol= 1e-5;
RMSE=zeros(1,maxiter);

for iter = 1:maxiter
    % Singular Value Thresholding Operator
    [U, S, V] = svd(Y, 'econ');
    S = soft_threshold(S, lambda);
    L = U*S*V';
    % Update S using the soft-thresholding operator
    S = soft_threshold(Y - L, 1);
    % Update Y by subtracting the difference between X and L + S from Y
    Y = Y + X_Omega - L - S
    % Calculate RMSE
    RMSE(iter) = sqrt(sum(sum((L - X).^2.*Omega))/num_remove);
    % Check for convergence
    if iter > 1 && abs(RMSE(iter)) < tol
        break;
    end
end

%find min iteration achieve RMSE approximate to 0

dif = abs(RMSE-tol);
miniter = find(dif == min(dif));
if RMSE(miniter) < tol
    disp(['matrix ', num2str(Xx_size), 'X', num2str(Xy_size), ' with rank',num2str(Xrank)])
    disp(['Minimum iteration: ' num2str(min(miniter))])
else
    disp(['Could not find minimum iteration for ',num2str(missingper),'%. RMSE values may increase over time.']);
    miniter = length(RMSE);
end

%plot graph
x = 1:maxiter;
figure;
plot(x, RMSE);
xlabel('No. of Iteration');
ylabel('RMSE');
title(['SVT Algorithm with ' num2str(missingper) '% missing elements']);

% Functions used in the Singular Value Thresholding Algorithm
function S = soft_threshold(X, lambda)
    S = sign(X).*max(abs(X) - lambda, 0);
end