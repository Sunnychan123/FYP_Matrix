function [L, S] = inexact_alm_rpca(M,lambda,tol,maxIter)

if nargin < 2
    lambda = 1/sqrt(max(size(M)));
end

if nargin < 3
    tol = 1e-7;
end

if nargin < 4
    maxIter = 1000;
end

[m,n] = size(M);

L = zeros(m,n);
S = zeros(m,n);
Y = zeros(m,n);

mu = 1.25/norm(M,'fro');

for iter = 1:maxIter
    % update L
    [U,S,V] = svd(M - S + (1/mu)*Y,'econ');
    diagS = diag(S);
    svp = length(find(diagS > 1/mu*lambda));
    if svp >= 1
        diagS = diagS(1:svp)-1/mu*lambda;
    else
        svp = 1;
        diagS = 0;
    end
    
    L = U(:,1:svp)*diag(diagS)*V(:,1:svp)';
    
    % update S
    S = max(M - L + (1/mu)*Y - lambda/mu,0) + min(M - L + (1/mu)*Y + lambda/mu,0);
    
    % update Y
    Y = Y + mu*(M - L - S);
    
    % update mu
    mu = min(mu*1.25,1e7);
    
    % check convergence
    if norm(M - L - S,'fro')/norm(M,'fro') < tol
        break;
    end
end

end
% 
% [m,n] = size(D);
% 
% if(nargin < 2) lambda = 1 / sqrt(m); end
% if(nargin < 3) tol = 1e-7; elseif(tol == -1) tol = 1e-7; end
% if(nargin < 4) maxIter = 1000; elseif(maxIter == -1) maxIter = 1000; end
% 
% % initialize
% Y = D;
% norm_two = norm(Y, 2);
% norm_inf = norm( Y(:), inf) / lambda;
% dual_norm = max(norm_two, norm_inf);
% Y = Y / dual_norm;
% 
% A_hat = zeros( m, n);
% E_hat = zeros( m, n);
% mu = 1.25/norm_two; % this one can be tuned
% mu_bar = mu * 1e7;
% rho = 1.5;          % this one can be tuned
% d_norm = norm(D, 'fro');
% 
% iter = 0;
% total_svd = 0;
% converged = false;
% stopCriterion = 1;
% sv = 10;
% while ~converged       
%     iter = iter + 1;
%     
%     temp_T = D - A_hat + (1/mu)*Y;
%     E_hat = max(temp_T - lambda/mu, 0);
%     E_hat = E_hat+min(temp_T + lambda/mu, 0);
% 
%     %[U,S,V] = svd(D - E_hat + (1/mu)*Y, 'econ');
%     [U,S,V] = svdecon(D - E_hat + (1/mu)*Y); % fastest
%     
%     diagS = diag(S);
%     svp = length(find(diagS > 1/mu));
%     if svp < sv
%         sv = min(svp + 1, n);
%     else
%         sv = min(svp + round(0.05*n), n);
%     end
%     
%     A_hat = U(:, 1:svp) * diag(diagS(1:svp) - 1/mu) * V(:, 1:svp)';    
% 
%     total_svd = total_svd + 1;
%     
%     Z = D - A_hat - E_hat;
%     
%     Y = Y + mu*Z;
%     mu = min(mu*rho, mu_bar);
%         
%     %% stop Criterion    
%     stopCriterion = norm(Z, 'fro') / d_norm;
%     if stopCriterion < tol
%         converged = true;
%     end    
%     
%     if mod( total_svd, 10) == 0
%         disp(['#svd ' num2str(total_svd) ' r(A) ' num2str(rank(A_hat))...
%             ' |E|_0 ' num2str(length(find(abs(E_hat)>0)))...
%             ' stopCriterion ' num2str(stopCriterion)]);
%     end
%     
%     if ~converged && iter >= maxIter
%         disp('Maximum iterations reached') ;
%         converged = 1 ;       
%     end
% end