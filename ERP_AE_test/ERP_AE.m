%% Classifying ERP via Autoencoders
%
% 2018-04-13 first release
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


%% Autoencoders classification

% split data and labels
XTrain = trainSet(2:end, :);
YTrain = trainSet(1, :);
XTest = testSet(2:end, :);
YTest = testSet(1, :);


% train classifier model
hiddenSize = 30; % number of neurons 30
autoenc = trainAutoencoder( XTrain, hiddenSize, ...
	'L2WeightRegularization', 0.0005, ... % 0.0005
	'SparsityRegularization', 0.5, ... % 0.5
	'SparsityProportion', 0.03, ... % 0.03
	'EncoderTransferFunction', 'logsig', ... % logsig | satlin
	'DecoderTransferFunction', 'logsig', ... % logsig | satlin | purelin
	'ScaleData', true, ... % true | false
	'MaxEpochs', 100, ... % 100
	'UseGPU', true ... % true | false
	);
features = encode(autoenc, XTrain);
softnet = trainSoftmaxLayer(features, YTrain, 'MaxEpochs', 500); % 500
netStack = stack(autoenc, softnet);
net = train(netStack, XTrain, YTrain, 'useGPU', 'yes');

% test classifier model
YTrain_ = net(XTrain);
YTest_ = net(XTest);

% round and transpose to [N, 1]
YTrain = round(YTrain)';
YTest = round(YTest)';
YTrain_ = round(YTrain_)';
YTest_ = round(YTest_)';

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







