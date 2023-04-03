clc
% Step1 generate X = U*V.'
mn=100;
rk=2;
U = randn(mn,rk);
Urank = rank(U);
V = randn(mn,rk);
Vrank = rank(V);
X = U*V.';
Xrank=rank(X);
[Xx_size, Xy_size]=size(X);
A=1:numel(X);
rem10per =floor(0.9*numel(X));
maxiter=100;
RMSE=zeros(1,maxiter);

% specify missing percentage
missingper = 10;

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

%NNM
lambda = 1/sqrt(max(Xx_size, Xy_size));
X_nnm = X;
for i = 1:maxiter
    X_nnm_old = X_nnm;
    
    % Use a randomized SVD algorithm
    [U,S,V] = rsvd(X_nnm + Omega.*(X - X_nnm), Xrank);
    S_new = max(0, S - lambda);
    X_nnm = U*S_new*V';
    
    % Use a stopping criterion based on the change in the objective function
    obj_old = norm(X_nnm_old(Omega == 1) - X(Omega == 1), 'fro');
    obj_new = norm(X_nnm(Omega == 1) - X(Omega == 1), 'fro');
    if abs(obj_new-obj_old)/obj_old < 1e-5
        break;
    end
    
    RMSE(1,i)  = sqrt(mean((X_nnm(Omega == 0) - X(Omega == 0)).^2));
end

tol= 1e-5;
dif = abs(RMSE-tol);
miniter = find(dif == min(dif));
if RMSE(miniter) < tol
    disp(['matrix ', num2str(mn), 'X', num2str(mn), ' with rank',num2str(rk)])
    disp(['Minimum iteration: ' num2str(miniter)])
else
    disp(['Could not find minimum iteration for ',num2str(missingper),'%. RMSE values may increase over time.']);
end


%check input and output
X_nnm;
%plot graph
x = 1:i;
figure;
plot(x, RMSE(1:i));
xlabel('No. of Iteration');
ylabel('RMSE');
title(['NNM with ' num2str(missingper) '% missing elements']);

% function for singular value thresholding
function [U,S,V] = rsvd(X,k)
    [m,n] = size(X);
    p = min(2*k, min(m,n));
    Omega = randn(n,p);
    Y = X*Omega;
    [Q,~] = qr(Y,0);
    B = Q'*X;
    [U,S,V] = svd(B,'econ');
    U = Q*U(:,1:k);
    S = S(1:k,1:k);
    V = V(:,1:k);
end