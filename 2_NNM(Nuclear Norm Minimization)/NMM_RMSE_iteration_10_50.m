clc
% Step1 generate X = U*V.'
mn=1000;
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
RMSE=zeros(5,maxiter);

% specify five missing percentages
missingpers = [10 20 30 40 50];

%remove specified percentages of elements
for i=1:length(missingpers)
    missingper = missingpers(i);
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
    for j = 1:maxiter
        X_nnm = svd_thresholding(X_nnm + Omega.*(X - X_nnm), lambda);
        RMSE(i,j)  = sqrt(mean((X_nnm(Omega == 0) - X(Omega == 0)).^2));
    end
    
    tol= 1e-5;
    dif = abs(RMSE(i,:)-tol);
    miniter = find(dif == min(dif));
    if RMSE(miniter) < tol
        disp(['matrix ', num2str(mn), 'X', num2str(mn), ' with rank',num2str(rk)])
        disp(['Minimum iteration: ' num2str(miniter)])
    else
        disp(['Could not find minimum iteration for ',num2str(missingper),'%. RMSE values may increase over time.']);
    end
end

%check input and output
X_nnm;
%plot graph
x = 1:maxiter;
figure;
hold on;
for i=1:size(RMSE, 1)
    y = RMSE(i,:);
    missingper = missingpers(i);
    plot(x,y, 'DisplayName', [num2str(missingper) '%']);
end
hold off;
legend('show');
xlabel('No. of Iteration');
ylabel('RMSE');
title(['NNM with different missing percentages for ' num2str(mn) 'x' num2str(mn) ' matrix']);

% function for singular value thresholding
function X_new = svd_thresholding(X,lambda)
    [U,S,V] = svd(X,'econ');
    S_new = max(0, S - lambda);
    X_new = U*S_new*V';
end