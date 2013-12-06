%% Test Set

clear all;
load('eigfaces.mat');
display('Reading Images')
K=20;
imPath = 'BoostingData/test/face/';
imType = '*.pgm';
list = dir([imPath imType]);
image = double(imread([imPath list(1).name]));
testF = zeros(19^2, length(list));
Meig = 64; Neig = 64;
for n=1:length(list)
    [M,N] = size(image);
    image = double(imread([imPath list(n).name]));
    image = image - mean(image(:));
    image = image / norm(image(:));
    testF(:,n) = image(:);
    if(M~=Meig || N~=Neig)
        % resize all eigenfaces if needed
        k_eigfaces = zeros(M*N,K);
        for i=1:K
            E = imresize(reshape(eigfaces(:,K),64,64),[M N]);
            E = E - mean(E(:));
            n2 = norm(E(:));
            if(n2~=0)
                E = E/n2;
            end
            k_eigfaces(:,i) = E(:);
        end
        [Meig,Neig] = size(E);
    end
end
Wf = k_eigfaces'*testF; % each col contains coeffs for each face
%%
imPath = 'BoostingData/test/non-face/';
imType = '*.pgm';
list = dir([imPath imType]);
image = double(imread([imPath list(1).name]));
testNF = zeros(19^2, length(list));
for n=1:length(list)
    [M,N] = size(image);
    image = double(imread([imPath list(n).name]));
    image = image - mean(image(:));
    image = image / norm(image(:));
    testF(:,n) = image(:);
    if(M~=Meig || N~=Neig)
        % resize all eigenfaces if needed
        k_eigfaces = zeros(M*N,K);
        for i=1:K
            E = imresize(reshape(eigfaces(:,K),64,64),[M N]);
            E = E - mean(E(:));
            n2 = norm(E(:));
            if(n2~=0)
                E = E/n2;
            end
            k_eigfaces(:,i) = E(:);
        end
        [Meig,Neig] = size(E);
    end
end

Wnf = k_eigfaces'*testNF; % each col contains coeffs for each face
W = [Wf Wnf];
%%
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