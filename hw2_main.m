close all;
clear all;

%%
K = 15;
path1 = 'BoostingData/train/face/';
path2 = 'BoostingData/train/non-face/';

path3 = 'BoostingData/test/face/';
path4 = 'BoostingData/test/non-face/';

outfile1='trainingVectors.mat';
outfile2='testingVectors.mat';
outfile3='adaboost.mat';
%% Extract Features from Training Set Images
ece797_featureExtraction(K,path1,path2,outfile1);
%% Extract Features from Testing Set Images
ece797_featureExtraction(K,path3,path4,outfile2);
%% Train an Adabost Classifier
ece797_trainAdaboost(outfile1,outfile2,outfile3);

%%
display('Testing Trained Classifier on Test Feature Vectors')
load(outfile2);
load(outfile3);
set = [Wf Wnf];
[K, numF] = size(Wf);
numNF = size(Wnf,2);
sol = [ones(1,numF) -ones(1,numNF)];
Htest=0;

figure;
plot(set(1,sol==1),set(2,sol==1),'bx'); hold on;
plot(set(1,sol==-1),set(2,sol==-1),'rx');

for p=1:1%length(ht)
    if(side(h_dim(p))==-1)
        Hx = 2*(set(h_dim(p),:) > ht(p))-1; % faces are above threshold
    else
        Hx = 2*(set(h_dim(p),:) < ht(p))-1; % faces are below threshold
    end
    Htest = Htest + at(p)*Hx;
    class = sign(Htest);
    misst(p) = sum(logical((1-sol.*class)/2))/(numF+numNF);
end
missed = logical((1-sol.*Htest)/2);
plot(set(1,missed),set(2,missed),'go');
hold off;
xlabel('E_1'); ylabel('E_2'); title('training set');
legend('Face','Not Face','Error');
misst