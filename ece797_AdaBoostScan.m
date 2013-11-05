%% HW2 - Part2: Facial Recognition
%% INIT
close all;

%% Load eigenface
s = [32 64 48 96 128];
shape = vision.ShapeInserter('Shape','Rectangles', ...
                                'BorderColor','Custom', ...
                                'CustomBorderColor',uint8(255));
for sn = 1:length(s)
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
    image = imread([imPath list(n).name]);
    if(size(image)>2);
        image = squeeze(mean(image,3));
    end
    image = image - mean(image(:));
    image = image / norm(image(:));
    m=eigfaceScan_ren(E,image);
    
end
% figure, imshow(image*255); 
% shape = vision.ShapeInserter('Shape','Rectangles', ...
%                                 'BorderColor','Custom', ...
%                                 'CustomBorderColor',uint8([255 0 0]));
% c = [size(image,1) size(image,2)];
%
% [y,x] = find(min(m(:))==m);
%
%
% J = step(shape,image,uint32([540 60 ncols nrows]));
% figure; imshow(J);
%
