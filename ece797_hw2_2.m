%% HW2 - Part2: Facial Recognition
%% INIT
close all;

%% Load eigenface
s = [32 64 48 96 128];
shape = vision.ShapeInserter('Shape','Rectangles', ...
    'BorderColor','Custom', ...
    'CustomBorderColor',uint8(255));
for sn = 4:4%length(s)
    load('eigface.mat')
    eigface = imresize(eigface,[s(sn) s(sn)]);
    A=eigface-mean(eigface(:));
    E = A/norm(A(:));
    [nrows ncols] = size(E);
    
    imPath = 'group_photos/';
    imType = '*.jpg';
    list = dir([imPath imType]);
    n=1;
    % for n=1:length(list)
    
    % Load image and reduce to grayscale
    image = imread([imPath list(1).name]);
    if(size(image)>2);
        image = squeeze(mean(image,3));
    end
    
    % Scan
    %     m=faceScan_ren(E,image);
    A = image;
    [M_E, N_E] = size(E);
    area = (M_E*N_E);
    [M_A, N_A] = size(A);
    %first compute the integral image
    intA = cumsum(cumsum(A,1),2);
    intA2 = cumsum(cumsum(A.^2,1),2);
    %Now, at each pixel compute the mean
    patchmean = size(M_A-M_E+1,N_A-N_E+1);
    patchnorm = size(M_A-M_E+1,N_A-N_E+1);
    
    i=1;
    patchmean(1,1) = intA(M_E,N_E)/area;
    patchnorm(1,1) = sqrt(intA2(M_E,N_E));
    for j = 2:N_A - N_E +1
        a2 = intA(i+M_E-1,j-1);
        a4 = intA(i+M_E-1,j+N_E-1);
        patchmean(i,j) = (a4 - a2)/area;
        a2 = intA2(i+M_E-1,j-1);
        a4 = intA2(i+M_E-1,j+N_E-1);
        patchnorm(i,j) = sqrt(a4 - a2);
    end
    j=1;
    for i = 2:M_A - M_E +1
        a3 = intA(i-1,j+N_E-1);
        a4 = intA(i+M_E-1,j+N_E-1);
        patchmean(i,j) = (a4 - a3)/area;
        a3 = intA2(i-1,j+N_E-1);
        a4 = intA2(i+M_E-1,j+N_E-1);
        patchnorm(i,j) = sqrt(a4 - a3);
    end
    for i = 2:M_A - M_E +1
        for j = 2:N_A - N_E +1
            a1 = intA(i-1,j-1);
            a2 = intA(i+M_E-1,j-1);
            a3 = intA(i-1,j+N_E-1);
            a4 = intA(i+M_E-1,j+N_E-1);
            patchmean(i,j) = (a4 + a1 - a2 - a3)/area;          
            a1 = intA2(i-1,j-1);
            a2 = intA2(i+M_E-1,j-1);
            a3 = intA2(i-1,j+N_E-1);
            a4 = intA2(i+M_E-1,j+N_E-1);
            patchnorm(i,j) = sqrt(a4 + a1 - a2 - a3);
        end
    end
    convolved = conv2(double(A), double(fliplr(flipud(E))),'valid');
    sumE = sum(E(:));
    score = convolved - sumE*patchmean;
    score = score./patchnorm;
    %
    %     c = [s(sn) s(sn)];
    %     tmp = m;
    %     tmp(tmp<max(m(:))*0.6)=0;
    figure, imagesc(score);
    [y,x] = find(score>0.15);
    figure, imshow(image);
    figure, imshow(uint8(image));
    hold on;
    plot(x,y,'o'); hold off
end
