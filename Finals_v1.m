cd 'C:\Users\user\Desktop\Final Project'

% Load the training images and labels
imgTrain = imageDatastore('train', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
train_data = load('train\train_label.mat');

% Convert ground truth data to training data suitable for object detection
trainingData = objectDetectorTrainingData(train_data.gTruth);

% Specify the network layers (AlexNet)
layers = 'alexnet';

% Define training options
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 1e-4, ...
    'ExecutionEnvironment', 'cpu');

% Train the Faster R-CNN object detector
% detector = trainFasterRCNNObjectDetector(trainingData, layers, options);

% Load the test images and labels
test_data = load('test\test_label.mat');
testingData = objectDetectorTrainingData(test_data.gTruth);

% Initialize an array to store overlap ratios
overlapRatio = zeros(size(testingData, 1), 1);

% Loop through the test images and perform detection
for i = 1:size(testingData, 1)
    % Read the test image
    img = imread(testingData.imageFilename{i});
    
    % Get the ground truth bounding boxes for all objects
    bbox_g_goggle = testingData.goggle{i};
    bbox_g_fins = testingData.fins{i};
    bbox_g_wetsuit = testingData.wetsuit{i};
    
    % Perform object detection
    [bboxes, scores, labels] = detect(detector, img);
    
    % Initialize a temporary variable to store overlap ratios for current image
    temp_overlapRatio = [];
    
    % Calculate overlap ratios for detected objects against all ground truth boxes
    if ~isempty(bboxes)
        temp_overlapRatio_goggle = bboxOverlapRatio(bboxes, bbox_g_goggle);
        temp_overlapRatio_fins = bboxOverlapRatio(bboxes, bbox_g_fins);
        temp_overlapRatio_wetsuit = bboxOverlapRatio(bboxes, bbox_g_wetsuit);
        
        % Combine all overlap ratios
        temp_overlapRatio = [temp_overlapRatio_goggle(:); temp_overlapRatio_fins(:); temp_overlapRatio_wetsuit(:)];
        
        % Remove zero overlap ratios
        temp_overlapRatio(temp_overlapRatio == 0) = [];
        
        % Calculate the mean overlap ratio for the current image
        if ~isempty(temp_overlapRatio)
            overlapRatio(i) = mean(temp_overlapRatio);
        else
            overlapRatio(i) = 0;  
        end
        
        % Annotate detected objects on the image with their labels
        labels_str = cellstr(labels); % Convert labels to cell array of strings
        I = insertObjectAnnotation(img, 'rectangle', bboxes, labels_str);
        figure, imshow(I)
    else
        overlapRatio(i) = 0;
        figure, imshow(img)
    end
end

% Display the average overlap ratio
disp('Average overlap ratio:');
disp(mean(overlapRatio));
