%% HW2 - Part2: Facial Recognition

%% Load eigenface
close all
s = [32 48 64 96 128];
s = [96 128];
load('eigfaces.mat');
load('adaboost.mat');
K=length(ht);
facelist = [];

imPath = 'group_photos/';
imType = '*.jpg';
list = dir([imPath imType]);
n=1;
% for n=1:length(list)
image = imread([imPath list(n).name]);
if(size(image)>2);
    image = squeeze(mean(image,3));
end

for ss=1:length(s)
    sn = s(ss)
    
    m = zeros(size(image));
    mm = size(m);
    [P,Q] = size(image);
    k_eigfaces = zeros(sn,sn,K);
    W = zeros((P-sn+1),(Q-sn+1),K+1);
    for i=1:max(h_dim)
        % facescan each eigenface
        k_eigfaces(:,:,i) = imresize(reshape(eigfaces(:,i),64,64),[sn sn]);
        A=k_eigfaces(:,:,i)-mean(mean(k_eigfaces(:,:,i)));
        E = A/norm(A(:));
        score = faceScan_ren(E,image);
        W(:,:,i) = score;
        % maybe display each one
        %     figure, imagesc(score);
    end
    
    m = zeros((P-sn+1),(Q-sn+1));
    for i=1:2:size(W,1)
        for j=1:2:size(W,2)
            patch = image(i:i+sn-1,j:j+sn-1);
            patch = patch-mean(mean(patch));
            patch = patch/norm(patch(:));
            % generate weighted sum of eigenfaces
            net = 0;
            for k=1:K
                net = net+k_eigfaces(:,:,k)*W(i,j,k);
            end
            % calculate error
            err = sum(sum((patch - net).^2))/(sn*sn);
            W(i,j,end) = err;
            % use coefficients and error for adaboost
            
            Htest=0;
            for p=1:K
                if(side(h_dim(p))==1)
                    Hx = 2*(W(i,j,h_dim(p)) > ht(p))-1; % faces are above threshold
                else
                    Hx = 2*(W(i,j,h_dim(p)) < ht(p))-1; % faces are below threshold
                end
                Htest = Htest + at(p)*Hx;
                mm(i,j)= Htest;
                m(i,j) = sign(Htest);
            end
        end
    end
    
    [y,x] = find(m==1);
    
    for i=1:length(x)
        err = W(y(i),x(i),end);
        facelist = [facelist; [x(i) y(i) sn sn err]];
    end
end
%%

% sort in order of ascending error
facelist2 = facelist;
[Y I] = sort(facelist(:,end),'ascend');
facelist = facelist(I,:);

for i=1:min(200,size(facelist,1))
    if(facelist(i,5)~=-1)
        xi = facelist(i,1);
        yi = facelist(i,2);
        si = facelist(i,3);
        for j=1:i-1
            % Check proximity if it hasnt been removed
            if(facelist(j,5)~=-1)
                xj = facelist(j,1);
                yj = facelist(j,2);
                sj = facelist(j,3);
                % if face i is within face j remove face i
                if((xj-si<xi) && (xi<xj+sj)) && ((yj-si<yi) && (yi<yj+sj))
                    facelist(i,5) = -1;
%                     [xi yi xj yj]
                    [i,j]
                end
            end
        end
    end
end
%%
figure, imagesc(image); hold on;
for i=1:200%size(facelist,1)
    if(facelist(i,5)~=-1)
        drawRect_ren(facelist(i,1:2), facelist(i,3:4));
    end
end