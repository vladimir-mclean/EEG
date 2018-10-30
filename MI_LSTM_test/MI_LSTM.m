%% Classifying Motor Imagery via LSTM
%
% 2018-10-24 first release
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


%% Load data

% load and skip 1st row (header)
raw = csvread(['Datasets' filesep 'MI_20181022_124446.csv'], 1, 0);

% sort data by id column
[~, idx] = sort( raw(:, 1) ); % sort just the first column
sorted = raw(idx, :); % sort the whole matrix using the sort indices

% get cell_ID and EEG
cell_ID = sorted(:, 3);
EEG = sorted(:, 4:end);


%% Split data to epochs

% extract start/end of the epochs
idxLH = []; idxRest = []; idxRH = [];
for ii = 1 : 1 : numel(cell_ID) - 1
	tmp = cell_ID(ii+1) - cell_ID(ii);
	if tmp == 1 idxLH = [idxLH ii+1]; end
	if tmp == -1 idxLH = [idxLH ii]; end
	if tmp == 2 idxRest = [idxRest ii+1]; end
	if tmp == -2 idxRest = [idxRest ii]; end
	if tmp == 3 idxRH = [idxRH ii+1]; end
	if tmp == -3 idxRH = [idxRH ii]; end	
end
boundsLH = reshape(idxLH, 2, [])';
boundsRest = reshape(idxRest, 2, [])';
boundsRH = reshape(idxRH, 2, [])';

% frequency sampling of EEG, Hz
fs = 250;

% filtering the EEG
fOrder = 4; % Butterworth filter order
% bandstop (remove 50 Nz noise)
freqs = [45 55];
[b, a] = butter(fOrder, freqs ./ (fs/2), 'stop');
% bandpass (Alpha band)
freqsA = [7 13];
[bA, aA] = butter(fOrder, freqsA ./ (fs/2), 'bandpass');
% bandpass (Beta band)
freqsB = [16 24];
[bB, aB] = butter(fOrder, freqsB ./ (fs/2), 'bandpass');
% apply filtering
data = filtfilt(b, a, EEG);
dataA = filtfilt(bA, aA, data);
dataB = filtfilt(bB, aB, data);

% set len and step of subepoch inside epoch
subEpLen = 250; % samples
subEpStep = 200; % samples

% make subepochs with Rigth Hand imagery
epochsLH = {}; labelsLH = []; i = 0;
for ii = 1 : 1 : size(boundsLH, 1)
	xA = dataA(boundsLH(ii, 1):boundsLH(ii, 2), :);
	xB = dataB(boundsLH(ii, 1):boundsLH(ii, 2), :);
	for iii = 1 : subEpStep : size(xA, 1) - subEpLen
		i = i + 1;
		xxA = xA(iii : iii+subEpLen-1, :);
		xxB = xB(iii : iii+subEpLen-1, :);
		xx = zscore([xxA xxB]);
		epochsLH{i} = xx';
		labelsLH(i) = 0;
	end
end

% make subepochs with Rest imagery
epochsRest = {}; labelsRest = []; i = 0;
for ii = 1 : 1 : size(boundsRest, 1)
	xA = dataA(boundsRest(ii, 1):boundsRest(ii, 2), :);
	xB = dataB(boundsRest(ii, 1):boundsRest(ii, 2), :);
	for iii = 1 : subEpStep : size(xA, 1) - subEpLen
		i = i + 1;
		xxA = xA(iii : iii+subEpLen-1, :);
		xxB = xB(iii : iii+subEpLen-1, :);
		xx = zscore([xxA xxB]);
		epochsRest{i} = xx';
		labelsRest(i) = 1;
	end
end

% make subepochs with Left Hand imagery
epochsRH = {}; labelsRH = []; i = 0;
for ii = 1 : 1 : size(boundsRH, 1)
	xA = dataA(boundsRH(ii, 1):boundsRH(ii, 2), :);
	xB = dataB(boundsRH(ii, 1):boundsRH(ii, 2), :);
	for iii = 1 : subEpStep : size(xA, 1) - subEpLen
		i = i + 1;
		xxA = xA(iii : iii+subEpLen-1, :);
		xxB = xB(iii : iii+subEpLen-1, :);
		xx = zscore([xxA xxB]);
		epochsRH{i} = xx';
		labelsRH(i) = 2;
	end
end


%% Make dataset

% balancing classes
minClass = min( [numel(labelsLH) numel(labelsRest) numel(labelsRH)] );
dataset = {}; label = [];
% LH
rp = randperm(numel(labelsLH), minClass);
dataset = [dataset epochsLH(rp)];
label = [label labelsLH(rp)];
% Rest
rp = randperm(numel(labelsRest), minClass);
dataset = [dataset epochsRest(rp)];
label = [label labelsRest(rp)];
% RH
rp = randperm(numel(labelsRH), minClass);
dataset = [dataset epochsRH(rp)];
label = [label labelsRH(rp)];


% divide dataset to 3 sets (training, validation, test)
valueset = [0 1 2];
catnames = {'Left' 'Rest' 'Right'};
% indices to divide random with 70/15/15 proportion
[trainInd, valInd, testInd] = dividerand( numel(label) );
% train
XTrain = dataset(trainInd)';
YTrain = categorical( label(trainInd), valueset, catnames )';
% validation
XVal = dataset(valInd)';
YVal = categorical( label(valInd), valueset, catnames )';
% test
XTest = dataset(testInd)';
YTest = categorical( label(testInd), valueset, catnames )';

divisors(numel(XTrain))


%% LSTM classification

% net setup
inputSize = size(XTrain{1}, 1); % length of features vector
numHiddenUnits = 50; % number of neurons
numClasses = 3;

% net layers setup
layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
	];

% net options setup
maxEpochs = 30;
miniBatchSize = 7;
options = trainingOptions( ...
	'adam', ... 'sgdm' | 'rmsprop' | 'adam'
	...'InitialLearnRate', 0.001, ... % default value is 0.01 for the 'sgdm' solver
	...'LearnRateSchedule', 'piecewise', ... 'none' (default) | 'piecewise'
	...'LearnRateDropPeriod', 2, ...
	...'LearnRateDropFactor', 0.95, ... 0.1 (default) | scalar from 0 to 1
	...'L2Regularization', 0.0001, ... 0.0001 (default) | nonnegative scalar
	'GradientThreshold', 1, ...
	...'Momentum', 0.5, ... 0.9 (default) | scalar from 0 to 1
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle', 'once', ... % 'once' (default) | 'never' | 'every-epoch'
	'ValidationData', {XVal, YVal}, ...
	'ValidationFrequency', 5, ...
	'ValidationPatience', 100, ...
    'Verbose', 0, ...
	'VerboseFrequency', 1, ...
    'Plots', 'training-progress', ... 'none' (default) | 'training-progress'
    'ExecutionEnvironment', 'auto' ... 'auto' (default) | 'cpu' | 'gpu'
	);

% train net
net = trainNetwork(XTrain, YTrain, layers, options);

% test net
YTrain_ = classify(net, XTrain, 'MiniBatchSize', miniBatchSize);
YVal_ = classify(net, XVal, 'MiniBatchSize', miniBatchSize);
YTest_ = classify(net, XTest, 'MiniBatchSize', miniBatchSize);

% plot confusion matrix
close all;
cm = confusionchart(YTest, YTest_);
cm.Title = '[CM for test dataset]';
cm.Normalization = 'total-normalized';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';




