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
RMSE_all = zeros(5, maxiter);

% specify five missing percentages
missingpers = [10 20 30 40 50];

%remove specified percentages of elements
for i=1:length(missingpers)
    missingper = missingpers(i);
    %calculate number of elements to remove
    num_remove = floor((100-missingper)/100*numel(X));
   
    %random permutation of integers start from 10%
    omega= A(randperm(numel(A),num_remove));
    %matrix zero with Xsize
    Omega = zeros (Xx_size,Xy_size);
    %change value from 0 to 1 in Omega matrix
    Omega (omega) = 1;
    %Remove matrix X elements, store at X_Omega
    X_Omega = X.*Omega;
    
    %LP2 
    [M , RMSE]=LP2(X,X_Omega,Omega,Xrank, maxiter);
    RMSE_all(i,:) = RMSE;
    
    %find min iteration achieve RMSE equal to tolerance
    %tolerance = 1e-6
    tol= 1e-5;
    dif = abs(RMSE-tol);
    miniter = find(dif == min(dif));

    RMSE(miniter);
    if RMSE(miniter) <= tol % Change the if statement to check 
        % if the minimum RMSE is less than or equal to the tolerance value
        disp(['matrix ', num2str(mn), 'X', num2str(mn), ...
            ' with rank',num2str(rk),' missing',num2str(missingper),'%'])
        disp(['Minimum iteration: ' num2str(miniter)])
    else
        disp(['Could not find minimum iteration for ', ...
            num2str(missingper),'%. RMSE values may increase over time.'])
    end
end

%plot graph
x = 1:maxiter;
figure;
hold on;
for i=1:size(RMSE_all, 1)
    y = RMSE_all(i,:);
    missingper = missingpers(i);
    plot(x,y, 'DisplayName', [num2str(missingper) '%']);
end
hold off;
legend('show');
xlabel('No. of Iteration');
ylabel('RMSE');
title(['Least Square with different missing percentages for ' ...
    num2str(mn) 'x' num2str(mn) ' matrix']);