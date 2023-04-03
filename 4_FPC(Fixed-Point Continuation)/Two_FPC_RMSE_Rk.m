clc
% Step1 generate X = U*V.'
mn=50;
max_r = mn; % maximum rank to test
missing_pers = [10 20 30 40 50]; % missing percentages to test
n_missing_pers = length(missing_pers);
RMSE_all = zeros(n_missing_pers, max_r);
maxiter =100;


% Fixed-Point Continuation Algorithm
% initialize arrays for plotting rank vs. missing percentage and RMSE vs. iteration
rank_vals_fpc = [];
missing_perc_vals_fpc = [];
RMSE_vs_iter_fpc = [];
max_missing_perc_fpc = zeros(1,max_r); % array to store maximum missing percentage for each rank

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
        
        % Fixed-Point Continuation Algorithm
        lambda = 1/sqrt(max(Xx_size, Xy_size));
        L = zeros(size(X));
        S = zeros(size(X));
        mu = 1;
        for iter = 1:maxiter
            % Update L using SVD
            [U, S_diag, V] = svd(X_Omega - S);
            S_diag(lambda >= diag(S_diag)) = 0;
            L = U*S_diag*V';
            % Update S using soft-thresholding
            S = sign(X_Omega - L).*max(abs(X_Omega - L) - lambda/mu, 0);
            % Update mu
            mu = mu*1.1;
            % Calculate RMSE
            RMSE_all(i, r) = sqrt(sum(sum((X - L - S).^2))/numel(X));
            % Check convergence
            tol = 1e-6;
            if iter > 1 && r > 1 && abs(RMSE_all(i, r) - RMSE_all(i, r-1)) < tol
                break;
            end
        end
        % Update max_missing_perc_fpc
        max_missing_perc_fpc(r) = max(max_missing_perc_fpc(r), missingper);
    end
    % Update rank_vals_fpc and missing_perc_vals_fpc
    rank_vals_fpc = [rank_vals_fpc r*ones(1, n_missing_pers)];
    missing_perc_vals_fpc = [missing_perc_vals_fpc missing_pers];
    % Update RMSE_vs_iter_fpc
    RMSE_vs_iter_fpc = [RMSE_vs_iter_fpc RMSE_all(:, r)'];
end

% plot RMSE vs. rank for Fixed-Point Continuation Algorithm
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
title(['FPC with different rank for ' num2str(mn) 'x' num2str(mn), ' matrix']);