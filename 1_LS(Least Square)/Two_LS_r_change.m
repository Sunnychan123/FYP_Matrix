clc
% Step1 generate X = U*V.'
mn=20;
max_r = mn; % maximum rank to test
missing_pers = [10 20 30 40 50]; % missing percentages to test
n_missing_pers = length(missing_pers);
RMSE_all = zeros(n_missing_pers, max_r);
maxiter =100;
% initialize arrays
rank_vals = [];
missing_perc_vals = [];
RMSE_vs_iter = [];
%array to store maximum missing percentage for each rank
max_missing_perc = zeros(1,max_r); 

for r = 1:max_r
    U = randn(mn,r);
    Urank = rank(U);
    V = randn(mn,r);
    Vrank = rank(V);
    X = U*V.';
    Xrank=rank(X);
    [Xx_size, Xy_size]=size(X);
    A=1:numel(X);
    rem10per =floor(0.9*numel(X));
    
    for i=1:n_missing_pers
        missingper = missing_pers(i);
        %calculate number of elements to remove
        num_remove = floor((100-missingper)/100*numel(X));
        disp(['Rank ', num2str(r), ', missing percentage: ' num2str(missingper) '%']);
        
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
        RMSE_all(i,r) = RMSE(end);
        
        %find min iteration achieve RMSE equal to tolerance
        %tolerance = 1e-6
        tol= 1e-6;
        dif = abs(RMSE-tol);
        miniter = find(dif == min(dif));
        if RMSE(miniter) > tol 
            miniter = miniter+1;
            if miniter > maxiter
                miniter = maxiter;
            end
        end
    
        RMSE(miniter);
        if RMSE(miniter) <= tol % Change the if statement to check if 
            % the minimum RMSE is less than or equal to the tolerance value
            disp(['matrix ', num2str(mn), 'X', num2str(mn), ' with rank', ...
                num2str(r),' missing',num2str(missingper),'%'])
            disp(['Minimum iteration: ' num2str(miniter)])
        else
            disp(['Could not find minimum iteration for ', ...
                num2str(missingper),'%. RMSE values may increase over time.'])
        end
        
        % update max_missing_perc array if current missing percentage
        % is higher than previously stored value for this rank
        if missingper > max_missing_perc(r)
            max_missing_perc(r) = missingper;
        end
        
        %store values for plotting
        rank_vals = [rank_vals, r];
        missing_perc_vals = [missing_perc_vals, missingper];
        RMSE_vs_iter = [RMSE_vs_iter; RMSE];
    end
end

% plot RMSE vs. rank
figure;
hold on;
for i=1:n_missing_pers
    y = RMSE_all(i,:);
    missingper = missing_pers(i);
    plot(y, 'DisplayName', [num2str(missingper) '%']);
end
hold off;
legend('show');
xlabel('Rank');
ylabel('RMSE');
title(['RMSE vs. Rank with Matrix ' num2str(mn) 'x' num2str(mn)]);