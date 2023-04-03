clc
% Step1 generate X = U*V.'
mn=4;
rk=2;
U = randn(mn,rk)
V = randn(mn,rk)
X = U*V.'
Xrank=rank(X);
[Xx_size, Xy_size]=size(X)
A=1:numel(X)
rem10per =floor(0.9*numel(X))
maxiter=100;

% specify missing percentage
missingper = 30;

% remaining percentage
num_remain = floor((100-missingper)/100*numel(X))

%random permutation of integers for remove 30%
omega= A(randperm(numel(A),num_remain))
%matrix zero with Xsize
Omega = zeros (Xx_size,Xy_size)
%change value from 0 to 1 in Omega matrix
Omega (omega) = 1
%Remove matrix X elements, store at X_Omega
X_Omega = X.*Omega

%LP2 
[M , RMSE]=LP2(X,X_Omega,Omega,Xrank, maxiter);

%find min iteration achieve RMSE approximate to 0
%tolerance = 1e-5
tol= 1e-5;
dif = abs(RMSE-tol);
miniter = find(dif == min(dif));
if RMSE(miniter) < tol
    disp(['matrix ', num2str(mn), 'X', ...
        num2str(mn), ' with rank',num2str(rk)])
    disp(['Minimum iteration: ' num2str(miniter)])
else
    disp(['Could not find minimum iteration for ' ...
        ,num2str(missingper),'%. RMSE values may increase over time.']);
end


%check input and output
disp("generate")
%plot graph
x = 1:maxiter;
figure;
plot(x, RMSE);
xlabel('No. of Iteration');
ylabel('RMSE');
title(['Least Square with ' num2str(missingper) '% missing elements']);
