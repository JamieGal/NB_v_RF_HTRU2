% Naive Bayes script. Change variable "distType" to change the distribution type used or discretising values 
%
% max runtime (for distType='kde') approx 20 sec.


clear all
% clc
rng(46)

%Types: 'disc', 'kde', 'norm'
distType = 'norm';

%Read pre-prcoessed dataTrain into a table
data = readtable('HTRU_2.csv');

%If discretise values selected bin continuous values into n bins
if strcmp(distType,'disc')
    %Number of bins for each column of the data
    nBins = 10;
    %Set edge limits based on data values
    edges = linspace(min(data.meanIntegrated),max(data.meanIntegrated),nBins);
    %Discretise values
    data.meanIntegrated = discretize(data.meanIntegrated,edges);
    %Repeat for all other feature columns
    edges = linspace(min(data.stdIntegrated),max(data.stdIntegrated),nBins);
    data.stdIntegrated = discretize(data.stdIntegrated,edges);

    edges = linspace(min(data.kurtosisIntegrated),max(data.kurtosisIntegrated),nBins);
    data.kurtosisIntegrated = discretize(data.kurtosisIntegrated,edges);

    edges = linspace(min(data.skewnessIntegrated),max(data.skewnessIntegrated),nBins);
    data.skewnessIntegrated = discretize(data.skewnessIntegrated,edges);

    edges = linspace(min(data.meanDM_SNR),max(data.meanDM_SNR),nBins);
    data.meanDM_SNR = discretize(data.meanDM_SNR,edges);

    edges = linspace(min(data.stdDM_SNR),max(data.stdDM_SNR),nBins);
    data.stdDM_SNR = discretize(data.stdDM_SNR,edges);

    edges = linspace(min(data.kurtosisDM_SNR),max(data.kurtosisDM_SNR),nBins);
    data.kurtosisDM_SNR = discretize(data.kurtosisDM_SNR,edges);

    edges = linspace(min(data.skewnessDM_SNR),max(data.skewnessDM_SNR),nBins);
    data.skewnessDM_SNR = discretize(data.skewnessDM_SNR,edges);

end

%Split dataTrain into test and training for main training and test set
testTrainSplit = cvpartition(size(data,1),'HoldOut',0.3);
%Get indexes for test data
idx = testTrainSplit.test;
%Split based on these indexes
dataTrain = data(~idx,:);
dataTest  = data(idx,:);

%Set number of folds for cross validation
k=5;

%Create k fold partition
crossVal = cvpartition(size(dataTrain,1),'KFold',k);

%Create empty confusion matrix
confMat = [0 0 ; 0 0];

%Iterate through k folds
for i=1:k
    %Split training data into new training and validation set
    cvTrain = dataTrain(training(crossVal,i),:);
    cvTest = dataTrain(test(crossVal,i),:);
   
    %Fit model based on selection
    if strcmp(distType,'disc')
        %For discretised values, set features to categorical values
        mdl = fitcnb(cvTrain,'class','CategoricalPredictors',[1 2 3 4 5 6 7 8]);
    elseif strcmp(distType,'norm')
        mdl = fitcnb(cvTrain,'class','DistributionNames',{'normal' 'normal' 'normal' 'normal' 'normal' 'normal' 'normal' 'normal' });
    elseif strcmp(distType,'kde')
        mdl = fitcnb(cvTrain,'class','DistributionNames',{'kernel' 'kernel' 'kernel' 'kernel' 'kernel' 'kernel' 'kernel' 'kernel'});
    end
    %Get predictions for validation set based on the model trained
    cvPred = mdl.predict(cvTest);

    %Calculate classification error
    %Count number of matching values between predicted and real values
    count=0;
    %Iterate through values and compare real with predicted values for
    %validation set
    for j=1:size(cvPred,1)
        if cvPred(j)==cvTest.class(j)
            count = count + 1;
        end
    end
    %Calculate classification error and add to list to average at the end
    classAcc(i) = 1-(count/size(cvPred,1));
    %Calculate confusion matrix for validation set and sum with previous confusion
    %matrices to get overall matrix
    confMat = confMat + confusionmat(cvTest.class,cvPred);
    disp("Loss for fold " + i + ": " + classAcc(i));
end

%Plot confusion matrix
figure;
cm = confusionchart(confMat,[0 1]);
title("Confusion matrix for type " + distType);

%Average classification error over validation sets
disp("Cross validated loss average: " + mean(classAcc) + " with standard deviation: " + std(classAcc));

%Fit model based on all training data for selected model type
if strcmp(distType,'disc')
    mdl = fitcnb(dataTrain,'class','CategoricalPredictors',[1 2 3 4 5 6 7 8]);
elseif strcmp(distType,'norm')
    mdl = fitcnb(dataTrain,'class','DistributionNames',{'normal' 'normal' 'normal' 'normal' 'normal' 'normal' 'normal' 'normal' });
elseif strcmp(distType,'kde')
    mdl = fitcnb(dataTrain,'class','DistributionNames',{'kernel' 'kernel' 'kernel' 'kernel' 'kernel' 'kernel' 'kernel' 'kernel'});
end

%Get predictions for test set based on this model
[mdlPred,post,~] = mdl.predict(dataTest);

%Calculate classification error
%Count number of matching values between predicted and real values
count = 0;
%Iterate through values and compare real with predicted values for
%test set
for j=1:size(dataTest,1)
    if mdlPred(j)==dataTest.class(j)
        count = count + 1;
    end
end
%Calculate classification error and add to list to average at the end
mdlLoss = 1-(count/size(mdlPred,1));

[X,Y,T,AUC] = perfcurve(dataTest.class,post(:,2),1);
figure;
plot(X,Y);
title("ROC for Naive Bayes ("+distType+") Classifier")
xlabel('False positive rate') 
ylabel('True positive rate')

disp("Final model loss ("+distType+"): " + mdlLoss);
