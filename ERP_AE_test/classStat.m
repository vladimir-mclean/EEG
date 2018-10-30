function outdata = classStat(in_actual, in_predicted)

%
% (c) vb 2017-03-25
% 2018-09-23 add MCC metrics
%
% Statistic for classification result
%
% input:
% in_actual - vector of true responses
% in_predicted - vector of predicted responses
%
% output:
% outdata - structure
% ACC, PPV, TPR, FPR, TNR, FNR, F1Score, nullER, kappa, MCC
% 1    2    3    4    5    6    7        8       9      10
%

if size(in_actual, 2) ~= 1 || size(in_predicted, 2) ~= 1
	error ('Inputs must be a vector [N,1]');
end

if size(in_actual, 1) ~= size(in_predicted, 1)
	error ('Inputs must be the same size');
end


%
% confusionmat(group,grouphat) returns the confusion matrix C
% determined by the known group (group) and predicted group (grouphat)
%               Predicted Classes
%                    p'    n'
%              ___|_____|_____|
%       Actual  p |  TP |  FN |
%      Classes  n |  FP |  TN |
%
% Accuracy (ACC)
% ACC = (TP + TN) / (TP + FP + FN + TN)
% How often is the classifier correct?
%
% Precision or positive predictive value (PPV)
% PPV = TP / (TP + FP)
% When it predicts YES, how often is it correct?
%
% Sensitivity, recall, hit rate, or true positive rate (TPR)
% TPR = TP / (TP + FN)
% When actually YES, how often does it predict YES?
%
% Fall-out or false positive rate (FPR)
% FPR = FP / (FP + TN)
% When actually NO, how often does it predict YES?
%
% Specificity, selectivity or true negative rate (TNR)
% TNR = TN / (FP + TN)
% When actually NO, how often does it predict NO?
%
% Miss rate or false negative rate (FNR)
% FNR = FN / (FN + TP)
% When actually YES, how often does it predict NO?
%
% F1 score
% F1Score = 2*TP / (2*TP + FP + FN)
% Weighted average of the Sensitivity and Precision
%
% Null Error Rate
% nullER = (TN + FP) / (TP + FP + FN + TN)
% How often would be wrong if always predicted the majority class
%
% Cohen's Kappa
% This is essentially a measure of how well the classifier performed
% as compared to how well it would have performed simply by chance.
% In other words, a model will have a high Kappa score if there is
% a big difference between the accuracy and the null error rate.
% kappa = (ACC - E) / (1 - E) where E is:
% E = (A + B) / (TP + FP + FN + TN) where is A and B are:
% A = (TP + FN)*(TP + FP) / (TP + FP + FN + TN)
% B = (FP + TN)*(FN + TN) / (TP + FP + FN + TN)
%
% Matthews correlation coefficient (MCC)
% MCC = (TP*TN - FP*FN) / sqrt( (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) )
% The MCC is in essence a correlation coefficient between
% the true and predicted binary classifications.
% It returns a value between ?1 and +1. A coefficient of +1 represents
% a perfect prediction, 0 no better than random prediction
% and ?1 indicates total disagreement between prediction and observation.
%

C = confusionmat(in_actual, in_predicted, 'order', [1 0]);
TP = C(1,1); FN = C(1,2); FP = C(2,1); TN = C(2,2);

ACC = (TP + TN) / (TP + FP + FN + TN);
PPV = TP / (TP + FP);
TPR = TP / (TP + FN);
FPR = FP / (FP + TN);
TNR = TN / (FP + TN);
FNR = FN / (FN + TP);
F1Score = 2*TP / (2*TP + FP + FN);
nullER = (TN + FP) / (TP + FP + FN + TN);
A = (TP + FN)*(TP + FP) / (TP + FP + FN + TN);
B = (FP + TN)*(FN + TN) / (TP + FP + FN + TN);
E = (A + B) / (TP + FP + FN + TN);
kappa = (ACC - E) / (1 - E);
MCC = (TP*TN - FP*FN) / sqrt( (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) );

outdata = struct('ACC',ACC, 'PPV',PPV, 'TPR',TPR, ...
	'FPR',FPR, 'TNR',TNR, 'FNR',FNR, 'F1Score',F1Score, ...
	'nullER',nullER, 'kappa',kappa, 'MCC',MCC);

end




