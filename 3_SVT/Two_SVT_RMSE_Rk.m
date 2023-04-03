clc
% Step1 generate X = U*V.'
mn=50;
max_r = mn; % maximum rank to test
missing_pers = [10 20 30 40 50]; % missing percentages to test
n_missing_pers = length(missing_pers);
RMSE_all = zeros(n_missing_pers, max_r);
maxiter =100;
% initialize arrays for plotting rank vs. missing percentage and RMSE vs. iteration
rank_vals = [];
missing_perc_vals = [];
RMSE_vs_iter = [];
max_missing_perc = zeros(1,max_r); % array to store maximum missing percentage for each rank
tol = 1e-5;

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
        
        % Singular Value Thresholding Algorithm
        lambda = 1/sqrt(max(Xx_size, Xy_size));
        L = zeros(size(X));
        S = zeros(size(X));
        Y = X_Omega;
        for iter = 1:maxiter
             % Update L using SVD
            [U, S, V] = svd(Y, 'econ');
            S = soft_threshold(S, lambda);
            L = U*S*V';
            % Update S using the soft-thresholding operator
            S = soft_threshold(Y - L, 1);
            % Update Y by subtracting the difference between X and L + S from Y
            Y = Y + X_Omega - L - S;

            % Calculate RMSE
            RMSE_all(i, r) = sqrt(sum(sum((X - L - S).^2))/numel(X));
            % Check convergence
            if iter > 1 && r > 1 && abs(RMSE_all(i, r) - RMSE_all(i, r-1)) < tol
                break;
            end
        end
        % Update max_missing_perc
        max_missing_perc(r) = max(max_missing_perc(r), missingper);
    end
    % Update rank_vals and missing_perc_vals
    rank_vals = [rank_vals r*ones(1, n_missing_pers)];
    missing_perc_vals = [missing_perc_vals missing_pers];
    % Update RMSE_vs_iter
    RMSE_vs_iter = [RMSE_vs_iter RMSE_all(:, r)'];
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
title(['SVT with different rank for ', num2str(mn), 'x', num2str(mn), ' matrix']);

% Functions used in the Singular Value Thresholding Algorithm
function S = soft_threshold(X, lambda)
    S = sign(X).*max(abs(X) - lambda, 0);
end