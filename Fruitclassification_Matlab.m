
%% clearing all workspace and instances
close all;
clear;
clc;

%% Unzip dataset and create datastore
dataFolder = 'MY_data';
imds = imageDatastore(fullfile(dataFolder,"train"), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsTest = imageDatastore(fullfile(dataFolder,"test"), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% Count the classes in the dataset
classCount = countEachLabel(imds)

%%display image size
img=readimage(imds,1);
imageSize=size(img)

%% Split training dataset in 7:3 ratio for training, validation
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized'); 

%%display some sample images from train dataset from split

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
 subplot(4,4,i)
 I = readimage(imdsTrain,idx(i));
 imshow(I)
end

%% Load pretrained GoogLeNet
net = googlenet;

%% Modify the network for trans fer learning
% Create a layer graph from the network
lgraph = layerGraph(net);

% Number of classes
numClasses = numel(categories(imdsTrain.Labels));

% Replace the final layers
newFcLayer = fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
newSoftmaxLayer = softmaxLayer('Name', 'new_softmax');
newClassLayer = classificationLayer('Name', 'new_classoutput');

% Remove layers that need to be replaced
lgraph = replaceLayer(lgraph, 'loss3-classifier', newFcLayer);
lgraph = replaceLayer(lgraph, 'prob', newSoftmaxLayer);
lgraph = replaceLayer(lgraph, 'output', newClassLayer);

%% Data Augmentation
inputSize = net.Layers(1).InputSize;

augmenter = imageDataAugmenter( ...
    'RandRotation', [-10 10], ...
    'RandXTranslation', [-3 3], ...
    'RandYTranslation', [-3 3]);

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenter, 'ColorPreprocessing', 'gray2rgb');
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, ...
    'ColorPreprocessing', 'gray2rgb');
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

%% Set training options
options = trainingOptions('adam', ...
    'MiniBatchSize', 10, ...
    'MaxEpochs', 5, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 10, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% Train the network with the modified layers
[netTransfer, trainingInfo] = trainNetwork(augimdsTrain, lgraph, options);

%% Evaluate on test set
[YPred, scores] = classify(netTransfer, augimdsTest);

%% Calculate accuracy
YTest = imdsTest.Labels;
testAccuracy = mean(YPred == YTest) * 100;
disp("Test set accuracy: " + testAccuracy + "%");

%% Plot accuracy curves
figure;
hold on;
plot(1:length(trainingInfo.TrainingAccuracy), trainingInfo.TrainingAccuracy, 'r', 'LineWidth', 2);
xlabel('Epoch');
ylabel('Accuracy (%)');
legend('Training Accuracy', 'Location', 'southeast');
title('Training accuracy');
xlim([1 options.MaxEpochs]); % Set x-axis limits to the number of epochs
grid on;
hold off;

%% Plot confusion matrix
figure;
plotconfusion(YTest, YPred);
title('Confusion Matrix for Test Set');

%% Display some test results
numTestImages = numel(imdsTest.Labels);
rand = randperm(numTestImages,4);
figure
for j = 1:4
    subplot(2, 2, j);
    rI = readimage(imdsTest, rand(j));
    imshow(rI);
    title(sprintf('Predicted: %s', string(YPred(rand(j)))));
end

