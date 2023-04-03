% create low rank matrix and add Gaussian noise
n = 1000; % matrix size
r = 5; % rank of the low rank matrix
X = randn(n,r)*randn(r,n); % create low rank matrix
noise = 0.1*randn(n,n); % add Gaussian noise
Y = X + noise; % corrupted matrix

% recover low rank matrix using least squares method
X_hat = pinv(Y)*X;

% plot original, corrupted, and reconstructed matrices
figure;
subplot(1,3,1); imagesc(X); title('Original');
subplot(1,3,2); imagesc(Y); title('Corrupted');
subplot(1,3,3); imagesc(X_hat); title('Reconstructed');