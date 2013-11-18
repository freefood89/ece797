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
    
    score = faceScan_ren(E,image);
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
