%% Classifying ERP via SVM
%
% 2018-04-04 first release
%
%
%==========================================================================


%% Init

clearvars;
format compact;
close all;

% change the current folder to the folder of this m-file
if(~isdeployed)
	cd(fileparts(which(mfilename)));
end

% add current folder and all sub-folders to matlab path
[dirMfile, ~, ~] = fileparts( strcat(mfilename('fullpath'),'.m') );
addpath( genpath(dirMfile) );


%% Load datasets

trainSet = csvread(['Datasets' filesep 'train.csv']);
testSet = csvread(['Datasets' filesep 'test.csv']);


%% SVM classification

% split data and labels
XTrain = trainSet(2:end, :)';
YTrain = trainSet(1, :)';
XTest = testSet(2:end, :)';
YTest = testSet(1, :)';

% train classifier model
SVMModel = fitcsvm(XTrain, YTrain, ...
		'KernelFunction', 'linear', ... 'linear' (default) | 'gaussian' | 'rbf' | 'polynomial'
		...'PolynomialOrder', 3, ... 3 (default)
		'Solver', 'SMO', ... 'ISDA' | 'L1QP' | 'SMO'
		'IterationLimit', 100000, ... 1000000 (default)
		'Standardize', true, ...
		'Verbose', 1);

% test classifier model
YTrain_ = predict(SVMModel, XTrain);
YTest_ = predict(SVMModel, XTest);

% store res to struct
res(1) = classStat( YTrain_, YTrain );
res(2) = classStat( YTest_, YTest );

% disp classifier statistics
kappa1 = num2str( getfield(res, {1}, 'kappa', {1}), '%.3f' );
kappa2 = num2str( getfield(res, {2}, 'kappa', {1}),'%.3f' );
accu1 = num2str( getfield(res, {1}, 'ACC', {1}), '%.3f' );
accu2 = num2str( getfield(res, {2}, 'ACC', {1}), '%.3f' );
disp(' ');
disp( ['[TRAIN]: kappa ' kappa1 ', accu: ' accu1] );
disp( ['[TEST ]: kappa ' kappa2 ', accu: ' accu2] );
disp(' ');

% plot confusion matrix
c = confusionmat(YTest, YTest_');
close all;
figure('ToolBar', 'none', ...
	'Units', 'pixels', ...
	'Position', [300 300 500 500]);
plotConfMat(c, {'NonTarget stim', 'Target stim'});




