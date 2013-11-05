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
    
    m = zeros(size(image));
    [P,Q] = size(image);
    [M,N] = size(E);
    for i = 1:(P-N)
        for j = 1:(Q-M)
            patch = double(histeq(uint8(image(i:i+N-1,j:j+M-1))));
            patch = imresize(patch,[64 64]);
            
            numNF = size(Wnf,2);
            sol = [ones(1,numF) -ones(1,numNF)];
            Htest=0;
            for p=1:length(ht)
                if(side(h_dim(p))==1)
                    Hx = 2*(set(h_dim(p),:) > ht(p))-1; % faces are above threshold
                else
                    Hx = 2*(set(h_dim(p),:) < ht(p))-1; % faces are below threshold
                end
                Htest = Htest + at(p)*Hx;
                class = sign(Htest);
                misst(p) = sum(logical((1-sol.*class)/2))/(numF+numNF);
            end
        end
    end
    %     m=eigfaceScan_ren(E,image);
    figure, imagesc(m);
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
