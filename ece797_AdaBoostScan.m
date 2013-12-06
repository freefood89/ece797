%% HW2 - Part2: Facial Recognition

%% Load eigenface
close all
s = [32 48 64 96 128];
% s = [96 128];
load('eigfaces.mat');
K=20;
facelist = [];
[nrows ncols] = size(E);

imPath = 'group_photos/';
imType = '*.jpg';
list = dir([imPath imType]);
n=3;
% for n=1:length(list)
image = imread([imPath list(n).name]);
if(size(image)>2);
    image = squeeze(mean(image,3));
end

for ss=1:length(s)
    sn = s(ss)
    
    m = zeros(size(image));
    [P,Q] = size(image);
    k_eigfaces = zeros(sn,sn,K);
    W = zeros((P-sn+1),(Q-sn+1),K+1);
    for i=1:K
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
    for i=1:10:size(W,1)
        for j=1:10:size(W,2)
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
            for p=1:length(ht)
                if(side(h_dim(p))==1)
                    Hx = 2*(W(i,j,h_dim(p)) > ht(p))-1; % faces are above threshold
                else
                    Hx = 2*(W(i,j,h_dim(p)) < ht(p))-1; % faces are below threshold
                end
                Htest = Htest + at(p)*Hx;
                m(i,j) = sign(Htest);
            end
        end
    end
    
    [y,x] = find(m==1);
    
    for i=1:length(x)
        err = W(x(i),y(i),end);
        facelist = [facelist; [x(i) y(i) sn sn err]];
    end
end
%%

% sort in order of ascending error
facelist = sort(facelist,5,'ascend');
for i=1:size(facelist,1)
    xi = facelist(i,1);
    yi = facelist(i,2);
    si = facelist(i,3);
    for j=1:i-1
        if(facelist(j,5)==-1)
            % don't check face j if it's already removed
            break;
        end
        xj = facelist(j,1);
        yj = facelist(j,2);
        sj = facelist(j,3);
        % if face i is within face j remove face i
        if((xj-si<xi) && (xi<xj+sj)) || ((yj-si<yi) && (yi<yj+sj))
            facelist(i,5) = -1;
            break;
        end
    end
end
%%
figure, imagesc(image); hold on;
for i=1:size(facelist,1);
%     if(facelist(i,5)~=-1)
        drawRect_ren(facelist(i,1:2), facelist(i,3:4));
%     end
end