clear all
clc
rng(46)

% Load data
data = table2array(readtable('HTRU_2.csv'));

%Split dataTrain into test and training
testTrainSplit = cvpartition(size(data,1),'HoldOut',0.3);
idx = testTrainSplit.test;

% Separate to training and test dataTrain
dataTrain = data(~idx,:);
dataTest  = data(idx,:);

% Split features and labels
dataTrainX = dataTrain(:,1:end-1);
dataTrainY = dataTrain(:,end);
dataTestX = dataTest(:,1:end-1);
dataTestY = dataTest(:,end);

k = 5;
crossVal = cvpartition(dataTrainY,'KFold',k);

%% Tuning using Bayesian Optimization
% Setting hyperparameters

maxMinLeafSize = 40;
maxNumTrees = 400;
maxNumFeatures = size(dataTrainX,2)-1;
minLeafSize = optimizableVariable('minLeafSize',[1,maxMinLeafSize],'Type','integer');
numFeatures = optimizableVariable('numFeatures',[1,maxNumFeatures],'Type','integer');
numTrees = optimizableVariable('numTrees',[1,maxNumTrees],'Type','integer');

hyperparametersRF = [minLeafSize; numFeatures; numTrees];

%% Misclassification error - Bayesian Optimisation - generate 10 calculations for numTrees, minLeafSize and numFeatures

summary_table_MCR = [];
best_accuracy_MCR = 0;
tic
for i = 1:10

    MCRresults = bayesopt(@(params)lossFmcr(params,dataTrainX,dataTrainY,crossVal),hyperparametersRF, 'MaxObjectiveEvaluations', 30, 'ExplorationRatio', 0.5, 'Verbose',1);
    MCRminLeafSize = MCRresults.XAtMinEstimatedObjective.minLeafSize;
    MCRnumFeatures = MCRresults.XAtMinEstimatedObjective.numFeatures;
    MCRnumTrees = MCRresults.XAtMinEstimatedObjective.numTrees;

    %Train RF model
    mdl_MCR = TreeBagger(MCRnumTrees,dataTrainX, dataTrainY,'method','classification','OOBPrediction','on','MinLeafSize',MCRminLeafSize,'NumPredictorsToSample', MCRnumFeatures);

    order = unique(dataTrainY); 
    % Confusion matrix
    confusion_mat_MCR = confusionmat(dataTrainY,(str2num(cell2mat(mdl_MCR.predict(dataTrainX)))),'order',order);

    % Performance metrics for training data using generated hyperaparamters
    [precision_MCR, accuracy_MCR, AUC_MCR] = performance_metrics(mdl_MCR,confusion_mat_MCR, dataTrainX, dataTrainY);

    % Best performing model on all the training data
    if accuracy_MCR > best_accuracy_MCR
        best_accuracy_MCR = accuracy_MCR;
        best_minLeafSize_MCR = MCRminLeafSize;
        best_numFeatures_MCR = MCRnumFeatures;
        best_numTrees_MCR = MCRnumTrees;

    end
    %Performance metrics for each Bayesian Optimisation
    summary_table_MCR = [summary_table_MCR; precision_MCR accuracy_MCR AUC_MCR];
end
RFtime = toc;

mean_precision_MCR = mean(summary_table_MCR(:,1));
mean_accuracy_MCR = mean(summary_table_MCR(:,2));
mean_AUC_MCR = mean(summary_table_MCR(:,3));


% Rerun best hyperparameters on training data prior to comparison with test
% data
best_mdl = TreeBagger(best_numTrees_MCR,dataTestX, dataTestY,'method','classification','OOBPrediction','on','MinLeafSize',best_minLeafSize_MCR,'NumPredictorsToSample',best_numFeatures_MCR);

% Confusion matrix
order = unique(dataTrainY);
mdlPred = str2num(cell2mat(best_mdl.predict(dataTrainX)));
best_confusion_mat_MCR = confusionmat(dataTrainY,(str2num(cell2mat(best_mdl.predict(dataTrainX)))),'order',order);

figure;
cm = confusionchart(dataTrainY, mdlPred)
disp(cm)

% Generate performance metrics for testing data using generated hyperaparamters
[best_precision_MCR,best_accuracy_MCR,best_AUC_MCR] = performance_metrics(best_mdl,best_confusion_mat_MCR,dataTrainX,dataTrainY);


function mcrloss = lossFmcr(params,In,Out,cvp)

classA = @(XTRAIN,YTRAIN,XTEST)(predict(TreeBagger(params.numTrees,XTRAIN,YTRAIN,'method','classification', 'OOBPrediction','on','MinLeafSize',params.minLeafSize, 'NumPredictorsToSample', params.numFeatures), XTEST));

mcrloss = crossval('mcr',In,Out,'predfun',classA,'partition',cvp);
end

function [precision, accuracy, specificity, AUC] = performance_metrics(mdl,confusion_mat,features,labels)

precision = confusion_mat(1)/(confusion_mat(1) + confusion_mat(2));
accuracy = (confusion_mat(1) + confusion_mat(4))/sum([confusion_mat(1),confusion_mat(2),confusion_mat(3),confusion_mat(4)]);
specificity = confusion_mat(4)/(confusion_mat(4) + confusion_mat(3));

fprintf('Precision %6.4f\n',precision);
fprintf('Accuracy %6.4f\n',accuracy);
fprintf('Specificity %6.4f\n',specificity);

%ROC curve
[label,scores,cost] = predict(mdl,features); %score is posterior probability

figure;
[X,Y,T,AUC] = perfcurve(labels,scores(:,2), 1);
plot(X,Y, 'LineWidth',2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
end

