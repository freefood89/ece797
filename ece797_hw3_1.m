%% HW3 EM
close all;
Hxy = imread('space-invaders.png');
Hxy = double(rgb2gray(Hxy));
Hxy_z = cell(1,4);

Hxy(100:end,:) = 0;
Pxy = Hxy/sum(sum(Hxy));

%% Initialize

Px2 = rand(1,90);
Py2 = rand(1,90);
Px1y1 = rand(600-90+1,400-90+1);

%% Update

Px2y2XY = conv2(conv2(Px1y1, Py2'),Px2);

