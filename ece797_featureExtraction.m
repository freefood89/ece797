function ece797_featureExtraction(K,facePath,nonfacePath,outfile)
%% Parameterize Images
% This script loads all training and testing images and parameterizes them
% using eigenfaces. The parameters are then saved as .mat files. The output
% file will have as many columns as faces and as many rows as the dimension
% of the parameter vector of each face plus one error parameter.

load('eigfaces.mat');
k_eigfaces = zeros(64^2,K);
for i=1:K
    E = reshape(eigfaces(:,i),64,64);
    E = E - mean(E(:));
    n2 = norm(E(:));
    if(n2~=0)
        E = E/n2;
    end
    k_eigfaces(:,i) = E(:);
end
imType = '*.pgm';
list = dir([facePath imType]);
F_hat = double(imread([facePath list(1).name]));
display(['Extracting Features from ' num2str(length(list)) ' Faces'])
F = zeros(19^2, length(list));
Meig = 64; Neig = 64;
for n=1:length(list)
    [M,N] = size(F_hat);
    F_hat = double(imread([facePath list(n).name]));
    F_hat = F_hat - mean(F_hat(:));
    F_hat = F_hat / norm(F_hat(:));
    F(:,n) = F_hat(:);
    if(M~=Meig || N~=Neig)
        % resize all eigenfaces if needed
        k_eigfaces = zeros(M*N,K);
        for i=1:K
            E = imresize(reshape(eigfaces(:,i),64,64),[M N]);
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
Wf = k_eigfaces'*F; % each col contains coeffs for each face
%%
imType = '*.pgm';
list = dir([nonfacePath imType]);
display(['Extracting Features from ' num2str(length(list)) ' Non Faces'])
NF_hat = double(imread([nonfacePath list(1).name]));
NF = zeros(19^2, length(list));
for n=1:length(list)
    [M,N] = size(NF_hat);
    NF_hat = double(imread([nonfacePath list(n).name]));
    NF_hat = NF_hat - mean(NF_hat(:));
    NF_hat = NF_hat / norm(NF_hat(:));
    NF(:,n) = NF_hat(:);
    if(M~=Meig || N~=Neig)
        % resize all eigenfaces if needed
        k_eigfaces = zeros(M*N,K);
        for i=1:K
            E = imresize(reshape(eigfaces(:,i),64,64),[M N]);
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

Wnf = k_eigfaces'*NF; % each col contains coeffs for each face
W = [Wf Wnf];
%%
display(['Obtaining ' num2str(size(W,2)) ' Reconstruction Errors'])
F = [F NF];
err = zeros(1,size(W,2));
for m=1:size(W,2)
    % Reconstruct images from training set w/ weighted sums of K eigfaces
    NF_hat = 0;
    for n=1:K
        NF_hat = NF_hat + W(n,m)*k_eigfaces(:,n);
    end
    err(m) = sum((NF_hat - F(:,m)).^2)/length(NF_hat);
end
%%
Wf=[Wf; err(1:size(Wf,2))];
Wnf=[Wnf; err(size(Wf,2)+1:end)];
save(outfile,'Wf','Wnf');
display(['Feature Vectors Saved to: ' outfile])
end