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
    m=faceScan_ren(E,image);
    
%     figure, imagesc(m); title([int2str(s(sn)) 'x' int2str(s(sn))]);
    c = [s(sn) s(sn)];
    tmp = m;
    tmp(tmp<max(m(:))*0.6)=0;
    figure, imagesc(tmp);
%     [y,x] = find(max(m(:))==m);
%     m(y:y+s(sn)-1,x:x+s(sn)-1) = -Inf;
%     [y2,x2] = find(max(m(:))==m);
%     m(x:x+s(sn)-1,y:y+s(sn)-1) = 0;
%     J = step(shape,image*255,uint32([x y c;x2 y2 c]));
%     figure; imshow(J); title([int2str(s(sn)) 'x' int2str(s(sn))]);
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
