clear all
clc
rng(46)

%Read pre-prcoessed dataTrain into a table
data = readtable('HTRU_2.csv');

%%%%%%%%%%%%%%  Discretise values for Naive Bayes data  %%%%%%%%%%%%%%%

%Create copy of original data
dataNB = data;

%Number of bins for each column of the data
nBins = 10;
%Set edge limits based on data values
edges = linspace(min(data.meanIntegrated),max(data.meanIntegrated),nBins);
%Discretise values
dataNB.meanIntegrated = discretize(data.meanIntegrated,edges);
%Repeat for all other feature columns
edges = linspace(min(data.stdIntegrated),max(data.stdIntegrated),nBins);
dataNB.stdIntegrated = discretize(data.stdIntegrated,edges);

edges = linspace(min(data.kurtosisIntegrated),max(data.kurtosisIntegrated),nBins);
dataNB.kurtosisIntegrated = discretize(data.kurtosisIntegrated,edges);

edges = linspace(min(data.skewnessIntegrated),max(data.skewnessIntegrated),nBins);
dataNB.skewnessIntegrated = discretize(data.skewnessIntegrated,edges);

edges = linspace(min(data.meanDM_SNR),max(data.meanDM_SNR),nBins);
dataNB.meanDM_SNR = discretize(data.meanDM_SNR,edges);

edges = linspace(min(data.stdDM_SNR),max(data.stdDM_SNR),nBins);
dataNB.stdDM_SNR = discretize(data.stdDM_SNR,edges);

edges = linspace(min(data.kurtosisDM_SNR),max(data.kurtosisDM_SNR),nBins);
dataNB.kurtosisDM_SNR = discretize(data.kurtosisDM_SNR,edges);

edges = linspace(min(data.skewnessDM_SNR),max(data.skewnessDM_SNR),nBins);
dataNB.skewnessDM_SNR = discretize(data.skewnessDM_SNR,edges);

%Split dataTrain into test and training for main training and test set
testTrainSplit = cvpartition(size(data,1),'HoldOut',0.3);
%Get indexes for test data
idx = testTrainSplit.test;
%Split based on these indexes
dataTestNB  = dataNB(idx,:);
dataTestRF  = data(idx,:);

%%%%%%%%%%%%  Load models and generate predictions from test data  %%%%%%%%%%%%%%%

%Get predictions for test set based on this model
load('mdlnb.mat');
[mdlNBPred,postNB,~] = mdl.predict(dataTestNB);
load("mdlrf.mat");
[mdlRFPred,postRF,~] = best_mdl.predict(dataTestRF);

mdlRFPred = str2num(cell2mat(mdlRFPred));

%Calculate classification error
%Count number of matching values between predicted and real values
count = 0;
%Iterate through values and compare real with predicted values for
%test set
for j=1:size(dataTestNB,1)
    if mdlNBPred(j)==dataTestNB.class(j)
        count = count + 1;
    end
end
%Calculate classification error and add to list to average at the end
mdlNBLoss = count/size(mdlNBPred,1);

count = 0;
%Iterate through values and compare real with predicted values for
%test set
for j=1:size(dataTestRF,1)
    if mdlRFPred(j)==dataTestRF.class(j)
        count = count + 1;
    end
end
%Calculate classification error and add to list to average at the end
mdlRFLoss = count/size(mdlRFPred,1);

[Xn,Yn,Tn,AUCn] = perfcurve(dataTestNB.class,postNB(:,2),1);
[Xr,Yr,Tr,AUCr] = perfcurve(dataTestRF.class,postRF(:,2),1);
figure;
hold on
plot(Xn,Yn,'DisplayName','Naive Bayes');
plot(Xr,Yr,'DisplayName','Random Forest');
title("ROC Comparison")
xlabel('False positive rate') 
ylabel('True positive rate')
legend()

%Plot confusion matrix
figure;
confMatNB = confusionmat(dataTestNB.class,mdlNBPred);
cmvNB = confusionchart(confMatNB,[0 1]);
title("Confusion matrix Naive Bayes");

%Plot confusion matrix
figure;
confMatRF = confusionmat(dataTestRF.class,mdlRFPred);
cmvRF = confusionchart(confMatRF,[0 1]);
title("Confusion matrix Random Forest");

disp("Final model accuracy for Naive Bayes: " + mdlNBLoss + " and G-mean: " + g_mean(confMatNB));
disp("Final model accuracy for Random Forest: " + mdlRFLoss + " and G-mean: " + g_mean(confMatRF));