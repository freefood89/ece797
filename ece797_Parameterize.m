%% Parameterize Images
% This script loads all training and testing images and parameterizes them
% using eigenfaces. The parameters are then saved as .mat files. The output
% file will have as many columns as faces and as many rows as the dimension
% of the parameter vector of each face plus one error parameter.
clear all;

%% Training Set
load('eigfaces.mat');
display('Reading Images')
imPath = 'BoostingData/train/face/';
imType = '*.pgm';
list = dir([imPath imType]);
image = double(imread([imPath list(1).name]));
len = length(eigfaces(:,1));
trainF = zeros(128^2, length(list));
for n=1:length(list)
    image = double(histeq(uint8(imresize(imread([imPath list(n).name]),[128 128]))));
    image = image - mean(image(:));
    image = image / norm(image(:));
    trainF(:,n) = image(:);
end

imPath = 'BoostingData/train/non-face/';
imType = '*.pgm';
list = dir([imPath imType]);
image = double(imread([imPath list(1).name]));
len = length(eigfaces(:,1));
trainNF = zeros(128^2, length(list));
for n=1:length(list)
    image = double(histeq(uint8(imresize(imread([imPath list(n).name]),[128 128]))));
    image = image - mean(image(:));
    norm2 = norm(image(:));
    if(norm2~=0)
        image = image / norm2;
    end
    trainNF(:,n) = image(:);
end

K=20;
k_eigfaces = zeros(size(trainF,1),K);
for n=1:K
    eigface = eigfaces(:,n);
    k_eigfaces(:,n) = eigface(:);
end

display('Parametrizing Images')
Wf = k_eigfaces'*trainF; % each col contains coeffs for each face
Wnf = k_eigfaces'*trainNF; % each col contains coeffs for each face
W = [Wf Wnf];
orig= [trainF trainNF];
err = zeros(1,size(trainF,2)+size(trainNF,2));
for m=1:size(trainF,2)+size(trainNF,2)
    % Reconstruct images from training set w/ weighted sums of K eigfaces
    image = 0;
    for n=1:K
        image = image + W(n,m)*k_eigfaces(:,n);
    end
    err(m) = sum((image - orig(:,m)).^2)/length(image);
end
Wf=[Wf; err(1:size(Wf,2))];
Wnf=[Wnf; err(size(Wf,2)+1:end)];

save('trainingVectors.mat','Wf','Wnf');

%% Test Set

clear all;
load('eigfaces.mat');
display('Reading Images')

imPath = 'BoostingData/test/face/';
imType = '*.pgm';
list = dir([imPath imType]);
image = double(imread([imPath list(1).name]));
len = length(eigfaces(:,1));
testF = zeros(128^2, length(list));
for n=1:length(list)
    image = double(histeq(uint8(imresize(imread([imPath list(n).name]),[128 128]))));
    image = image - mean(image(:));
    image = image / norm(image(:));
    testF(:,n) = image(:);
end

imPath = 'BoostingData/test/non-face/';
imType = '*.pgm';
list = dir([imPath imType]);
image = double(imread([imPath list(1).name]));
len = length(eigfaces(:,1));
testNF = zeros(128^2, length(list));
for n=1:length(list)
    image = double(histeq(uint8(imresize(imread([imPath list(n).name]),[128 128]))));
    image = image - mean(image(:));
    norm2 = norm(image(:));
    if(norm2~=0)
        image = image / norm2;
    end
    testNF(:,n) = image(:);
end

K=20;
k_eigfaces = zeros(size(testF,1),K);
for n=1:K
    eigface = eigfaces(:,n);
    k_eigfaces(:,n) = eigface(:);
end

Wf = k_eigfaces'*testF; % each col contains coeffs for each face
Wnf = k_eigfaces'*testNF; % each col contains coeffs for each face

W = [Wf Wnf];
orig= [testF testNF];
err = zeros(1,size(testF,2)+size(testNF,2));
for m=1:size(testF,2)+size(testNF,2)
    % Reconstruct images from training set w/ weighted sums of K eigfaces
    image = 0;
    for n=1:K
        image = image + W(n,m)*k_eigfaces(:,n);
    end
    err(m) = sum((image - orig(:,m)).^2)/length(image);
end
Wf=[Wf; err(1:size(Wf,2))];
Wnf=[Wnf; err(size(Wf,2)+1:end)];

save('testingVectors.mat','Wf','Wnf');