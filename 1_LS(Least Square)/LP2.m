function [Out_X,RMSE] = LP2(M,M_Omega,Omega_array,rak,maxiter)
% This matlab code implements the l2-reg 
% method for Matrix Completion.
%
% M - m x n observed matrix 
% Omega_array is the subset
% rak is the rank of the object matrix
% maxIter - maximum number of iterations
%[oldM , oldRMSE]=LP2(X,X_Omega,Omega,Xrank, maxiter);

[r,c] = size(M_Omega);
U = randn(r,rak);
V = randn(rak,c);
RMSE = zeros(1, maxiter);
for iter = 1 : maxiter
    for j = 1:c
        row = find(Omega_array(:,j) == 1);
        U_I =  U(row,:);
        b_I = M_Omega(row,j);
        V(:,j) = pinv(U_I)* b_I;
    end
    for i = 1 : r
        col = find(Omega_array(i,:) == 1);
        V_I = V(:,col);
        b_I = M_Omega(i,col);
        U(i,:) = b_I * pinv(V_I);
    end
    X = U*V;
    RMSE(1,iter) = norm(M- X,'fro')/norm(M,'fro');
end
    Out_X = X;
end