%% 18797 Generating an eigenface

%% Read in faces

imPath = 'lfw1000/';
imType = '*.pgm';
list = dir([imPath imType]);
image = double(imread([imPath list(1).name]));
[nrows,ncols] = size(image);
Y = zeros(128*128, length(list));
for n=1:length(list)
    image = double(histeq(uint8(imresize(imread([imPath list(n).name]),[128 128]))));
    image = image - mean(image(:));
    image = image / norm(image(:));
    [nrows,ncols] = size(image);
    Y(:,n) = image(:);
end


%% Compute Eigenface

[eigfaces,S,V] = svd(Y,0);
eigface = reshape(eigfaces(:,1),nrows,ncols);
figure, imagesc(eigface);

%% Scan Photos for Faces

save('eigface.mat','eigface');
save('eigfaces.mat','eigfaces');

