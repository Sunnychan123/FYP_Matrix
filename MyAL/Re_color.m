% Load the grayscale image
img_gray = imread('image.png');

% Load the pre-trained colorization network
net = load('colorization_net.mat');

% Resize the grayscale image to match the input size of the network
img_gray_resized = imresize(img_gray, net.inputSize(1:2));

% Convert the grayscale image to Lab color space
img_lab = rgb2lab(img_gray_resized);

% Split the Lab image into luminance and chrominance channels
img_lum = img_lab(:,:,1);
img_chrom = img_lab(:,:,2:3);

% Normalize the luminance channel to [-1, 1]
img_lum_norm = (img_lum - 50) / 50;

% Colorize the chrominance channels using the pre-trained network
img_color_norm = predict(net, img_lum_norm);

% Denormalize the chrominance channels to [0, 1]
img_color = (img_color_norm + 1) / 2;

% Combine the luminance and chrominance channels into a Lab image
img_lab_colorized = cat(3, img_lum, img_color * 128);

% Convert the Lab image to RGB color space
img_colorized = lab2rgb(img_lab_colorized);

% Display the colorized image
figure;
subplot(1,2,1);
imshow(img_gray);
title('Grayscale image');
subplot(1,2,2);
imshow(img_colorized);
title('Colorized image');