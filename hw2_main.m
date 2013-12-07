close all;
clear all;

%%
K = 20;
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
ece797_featureExtraction(K,3,path4,outfile2);
%% Train an Adabost Classifier
ece797_trainAdaboost(outfile1,outfile2,outfile3);
