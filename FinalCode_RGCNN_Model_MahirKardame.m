close all;                                                                  % Close all open figures and clears the workspace

% Specify the number of images to be used for individual rice grain class
numImages = 300;                                                            % Variable 'numImages' to value of 300 (300 images to be used per individual class, 1,500 images total)
fprintf('Using %d images for training, validation & testing.\n', numImages); % Prints message to the command window stating how many images are being used for training, validation & testing

% Loads the segemented rice grain image data
imds = imageDatastore('Rice_Image_Dataset', ...                             % This line creates an ImageDatastore object (`imds`) 
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');               % from the images in the "Rice_Image_Dataset" directory.

% Limits the number of images being used                                    % Checks if the number of image files in the ImageDatastore 
if numel(imds.Files) > numImages                                            % object `imds` is greater than the specified number of images (`numImages`).
    imds = splitEachLabel(imds, numImages, 'randomized');                   % Splits the ImageDatastore `imds` into subsets, ensuring that
end                                                                         % each class contains the specified number of images 
                                                                            % (`numImages`) and shuffles them randomly within each class.
% Splits image dataset into 3 ratios
[imdsTrain] = splitEachLabel (imds,0.7,'randomized');                       % Splits the `imds` into training subset (imdsTrain) with 70% of the data, shuffling the data randomly within each class
[imdsValidation] = splitEachLabel (imds,0.15,'randomized');                 % Splits the `imds` into validation subset (imdsValidation) with 15% of the data, shuffling the data randomly within each class.
[imdsTesting] = splitEachLabel (imds,0.15,'randomized');                    % Splits the `imds` into testing subset (imdsTesting) with 15% of the data, shuffling the data randomly within each class.

% Setting up Network Architecture
net = alexnet;                                                              % Loads a pre-trained AlexNet neural network model, assigning it to the variable `net`
inputSize = net.Layers(1).InputSize;                                        % Retrieves the input size of the first layer of the pre-trained neural network `net` and assigns it to the variable `inputSize`
layersTransfer = net.Layers(1:end-3);                                       % Extracts all layers of the pre-trained network `net` except for the last three layers, storing them in the variable `layersTransfer`
numClasses = numel(categories(imds.Labels));                                % Calculates the number of classes (unique labels) in the ImageDatastore `imds` by counting the number of unique categories in the labels of the ImageDatastore, and assigns this count to the variable `numClasses`

% Adding Layers from Networks structure
layers = [
    layerstransfer                                                          % Extracts transferable layers from a pre-trained neural network.
    % This line creates a fully connected layer with `numClasses` neurons, where `numClasses` is the number of classes in the dataset. Additionally, it sets the learning rate factors for the weights and biases of the layer to 20, which can influence how quickly these parameters are updated during training.
     fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, ...       % Adjusts the learning rate for updating weights and biases during training.
     'BiasLearnRateFactor',20)                                              % Adjusts the learning rate for updating biases during training.
     softmaxLayer                                                           % Creates a softmax layer as output layer
     classificationLayer];                                                  % Creates classification layer as final layer in Neural Network

% Defines the image augmentation parameters (Pre-Processes Images)
pixelRange = [-30 30];                                                      % Sets the range of pixel values for adjustments during image augmentation
rotationRange = [-15, 15];                                                  % Sets the range of rotation angles for adjustments during image augmentation.
scalingRange = [0.5, 1.5];                                                  % Sets the range of scaling factors for adjustments during image augmentation.
imageAugmenter = imageDataAugmenter( ...                                    % Creates an imageDataAugmenter object named `imageAugmenter` with specified augmentation parameters, such as reflection, translation, rotation, and scaling, used for augmenting image data during training.
    'RandXReflection', true, ...                                            % Specifies that random horizontal reflection should be applied during image augmentation, which flips images horizontally randomly to increase dataset variability.
    'RandXTranslation', pixelRange, ...                                     % Specifies that random horizontal translation should be applied during image augmentation, where `pixelRange` defines the range within which the images can be translated horizontally.
    'RandYTranslation', pixelRange, ...                                     % Specifies that random vertical translation should be applied during image augmentation, where `pixelRange` defines the range within which the images can be translated vertically.
    'RandRotation', rotationRange, ...                                      % Specifies that random rotation should be applied during image augmentation, where `rotationRange` defines the range within which the images can be rotated.
    'RandScale', scalingRange);                                             % Specifies that random scaling should be applied during image augmentation, where `scalingRange` defines the range within which the images can be scaled.                                       

% Create augmented image datastore
augimds = augmentedImageDatastore(inputSize(1:2), imds, ...                 % Creates an augmentedImageDatastore object named `augimds`, which applies the specified
    'DataAugmentation', imageAugmenter);                                    % image augmentation techniques (`imageAugmenter`) to the images in the original ImageDatastore `imds`, with a target input size defined by `inputSize(1:2)`.

% Setting Training parameters values for Optimal Combination Parameter
maxEpoch =[10, 20, 35];                                                     % Defines a list of maximum epochs for training, with values 10, 20, and 35.An epoch represents one complete pass through the entire training dataset during the training process.    
learningRates = [1e-4, 1e-5];                                               % Defines a list of Learning Rates for training which is  used as starting value for the learning rate in neural network training.
miniBatchSizes = [100, 150, 200];                                           % Defines a list of mini batch size for training.The mini-batch size determines the number of samples used in each iteration of training.
bestParams = struct('MaxEpochs', 0,'LearningRate', 0,'MiniBatchSize', 0);   % Creates a structure named `bestParams` with fields `'MaxEpochs'`, `'LearningRate'`, and `'MiniBatchSize'`, initialized to 0. These fields will be used to store the best parameters achieved during the training phase.
bestAccuracy = 0;                                                           % Initializes the variable `bestAccuracy` to 0. It will be used to track the highest accuracy achieved during training.
bestPrecision = 0;                                                          % Initializes the variable `bestPrecision` to 0. It will be used to track the highest precision achieved during training
bestTrainingTime = inf;                                                     % Initializes the variable `bestTrainingTime` to positive infinity (`Inf`). It will be used to track the shortest training time achieved during training.

% Initilaises loops for RGCNN Model to run/iterate through with different parameter combinations 
for maxEpochs = maxEpoch                                                    % Initiates a loop where `maxEpochs` iterates over the values in the `maxEpochsList`, allowing for training the neural network with different numbers of epochs.
    for learningRate = learningRates                                        % Initiates a loop where `learningRate` iterates over the values in the `learningRates`, allowing for training the neural network with different learning rates.
        for miniBatchSize = miniBatchSizes                                  % Initiates a loop where `miniBatchSize` iterates over the values in the `miniBatchSizes`, allowing for training the neural network with different mini-batch sizes.
            fprintf(['Training with Epochs=%d, LearningRate=%.1e,' ...      % Prints a formatted message to the console, indicating the current training configuration, including the number of epochs, learning rate, and mini-batch size.
                ' MiniBatchSize=%d\n'], ...                                 % Completes the formatted message by specifying the placeholder for the mini-batch size (`%d`) and adding a newline character at the end (`\n`).
                maxEpochs, learningRate, miniBatchSize);                    % Inserts the actual values of `maxEpochs`, `learningRate`, and `miniBatchSize` into the formatted message to be printed to the console.

             % Setting up Training Options with Variable Learning Rate
             options = trainingOptions('adam',...                           % Creates training options for the neural network using the Adam optimizer with specified parameters and settings for training.
              'MaxEpochs', maxEpochs, ...                                   % Sets the maximum number of epochs for training to the value specified by the variable `maxEpochs`.
                'InitialLearnRate', learningRate, ...                       % Sets the initial learning rate for training to the value specified by the variable `learningRate`.
                'MiniBatchSize', miniBatchSize, ...                         % Sets the mini-batch size for training to the value specified by the variable `miniBatchSize`.
                'Shuffle', 'every-epoch', ...                               % Specifies that the training data should be shuffled at the beginning of each epoch during training to ensure that the model does not learn from the same sequence of data samples every time.
                'ValidationData', augimds, ...                              % Sets the validation data for the training process to the augmented image datastore `augimds`, allowing the model's performance to be evaluated on a separate validation set during training.
                'ValidationFrequency', 3, ...                               % Specifies that validation should be performed every 3 epochs during training, allowing the model's performance to be evaluated periodically to monitor for overfitting and ensure generalization to unseen data.
                'Verbose', false, ...                                       % Sets the verbosity level during training to `false`, meaning that training progress and status updates will not be displayed in the MATLAB command window during the training process.
                'Plots', 'training-progress', ...                           % Specifies that training progress plots should be displayed during the training process, showing metrics such as loss and accuracy over epochs to visualize the model's performance.
                'L2Regularization', 0.01, ...  % Weight decay               % Helps prevent overfitting by penalizing large weights in the neural network during training.
                'LearnRateSchedule', 'piecewise', ...                       % Learning rate will be adjusted at specific intervals during training.
                'LearnRateDropPeriod', 2, ...                               % Drops learning rate to every 2 epochs during training.
                'LearnRateDropFactor', 0.1);                                % Drops learning rate by a factor of 0.1

               % Starts timer
               trainingStartTime = tic;                                     % Starts a timer (`tic`) to record the starting time of the training process.
               
              % Trains the RGCNN network
              netTransfer = trainNetwork(augimds, layers, options );        % Trains the neural network using the augmented image datastore, defined layers, and specified training options, and assigns the trained network to the variable `netTransfer`.

              % Stop timer and calculate total training time taken 
              trainingEndTime = toc(trainingStartTime);                     % Stops the timer (`toc`) and records the ending time of the training process, calculating the total training time.

              % Evaluates the trained network
              [YPred, scores] = classify(netTransfer, augimds);             % Classifies the augmented images using the trained neural network `netTransfer`, producing predictions (`YPred`) and scores for each class.
              YValidation = imds.Labels;                                    % Assigns the true labels of the original image datastore `imds` to the variable `YValidation`. These labels will be used for evaluation purposes.

              % Calculates overall accuracy for 5 rice classes
              accuracy = mean(YPred == YValidation);                        % Calculates the accuracy of the predictions by comparing the predicted labels (`YPred`) with the true labels (`YValidation`) and then taking the mean of the resulting logical array, where `1` indicates a correct prediction and `0` indicates an incorrect prediction.
              fprintf('Overall Accuracy: %.2f%%\n', accuracy * 100);        % Prints a formatted message to the console, indicating the overall accuracy of the predictions as a percentage, with two decimal places, using the calculated accuracy value (`accuracy`).
              
              % Calculates confusion matrix for detailed insights
              C = confusionmat(YValidation, YPred);                         % Calculates the confusion matrix `C` using the true labels `YValidation` and the predicted labels `YPred`.
              figure;                                                       % Creates a new figure window for plotting
              confusionchart(C, categories(imds.Labels));                   % Creates a confusion chart for the confusion matrix 'C' with category labels extracted from 'imds.Labels'.

               % Display randomised images with Accuracy % (Mathworks Code)
            idx = randperm(numel(imdsValidation.Files),6);                  % Selects 6 random rice grain images from the validation dataset stored in `imdsValidation.Files`.
            figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);      % Creates a figure with normalized units and sets its position to [0.2 0.2 0.4 0.4], specifying its location and size on the screen.
            for i = 1:numel(idx)                                            % Initiates a loop that iterates over each element in the array `idx`, running the loop body for each index value stored in `idx`.
                subplot(3,3,i)                                              % Sets up subplots within a 3x3 grid layout in the current figure, with the loop index `i` determining the position of each subplot.
                imshow(readimage(imdsValidation,idx(i)));                   % Displays an image from the validation dataset at the index `idx(i)` in a subplot grid.
                prob = num2str(100*max(scores(idx(i),:)),3);                % Calculates the maximum probability score from the `scores` matrix at index `idx(i)`, converts it to a percentage, and converts it to a string with three significant figures.
                predClass = char(YPred(idx(i)));                            % Converts the predicted class label at index `idx(i)` in `YPred` to a character array.
                title(predClass + ", " + prob + "%")                        % Sets the title of the current subplot to display the predicted class followed by the corresponding probability percentage.
            end

            % Calculates accuracy & precision for each grain type class (unique)
            grainLabels = unique(imds.Labels);                              % Creates an array 'grainLabels' containing unique labels extracted from 'imds.Labels'.
            numGrains = numel(grainLabels);     % 5 rice classes            % Calculates the number of unique grain labels stored in the 'grainLabels' array and assigns it to 'numGrains'.
            accuracyPerGrain = zeros(numGrains, 1);                         % Initializes a column vector 'accuracyPerGrain' filled with zeros, where each row corresponds to the accuracy of a unique grain label.
            precisionPerGrain = zeros(numGrains, 1);                        % Initializes a column vector 'precisionPerGrain' filled with zeros, where each row corresponds to the precision of a unique grain label.

            % Calculates Inidivudial Rice Grain accuracy in a loop
            for i = 1:numGrains                                             % Starts a loop that iterates over each unique grain label index, from 1 to 'numGrains'.
                 idx = find(imds.Labels == grainLabels(i));                  % Finds the indices of elements in 'imds.Labels' that match the current unique grain label indexed by 'i' and assigns them to 'idx'.
                accuracyPerGrain(i) = mean(YPred(idx) == YValidation(idx)); % Calculates the accuracy for the current unique grain label indexed by 'i' and assigns the mean accuracy to 'accuracyPerGrain(i)'.
                precisionPerGrain(i) = sum(YPred(idx) == YValidation(idx)) / sum(YPred == grainLabels(i)); % Calculates the precision for the current unique grain label indexed by 'i' and assigns it to 'precisionPerGrain(i)'.
                fprintf('%s: Accuracy = %.2f%%, Precision = %.2f%%\n', char(grainLabels(i)), accuracyPerGrain(i) * 100, precisionPerGrain(i) * 100); % Prints the accuracy and precision for the current unique grain label indexed by 'i' in a formatted string.
            end

             % Updates the best parameters combination and training times
             if accuracy > bestAccuracy                                     % Checks if the current accuracy is greater than the best accuracy.
                 bestAccuracy = accuracy;                                   % Updates the variable 'bestAccuracy' to the current accuracy value.
                 bestPrecision = mean(precisionPerGrain);                   % Calculates the mean precision across all grain labels and assigns it to 'bestPrecision'.
                 bestParams.MaxEpochs.maxEpochs;                            % Assigns the value of 'maxEpochs' to the field 'MaxEpochs' within the structure 'bestParams'.
                 bestParams.LearningRate = learningRate;                    % Assigns the value of 'learningRate' to the field 'LearningRate' within the structure 'bestParams'.
                 bestParams.MiniBatchSize = miniBatchSize;                  % Assigns the value of 'miniBatchSize' to the field 'MiniBatchSize' within the structure 'bestParams'.
                 bestTrainingTime = trainingEndTime;                        % Assigns the value of 'trainingEndTime' to the variable 'bestTrainingTime'.
             end

   % Stops model if average accuracy reaches 95%
            if accuracy >= 0.95                                             % Checks if the accuracy is greater than or equal to 95%.
                fprintf(['Target accuracy of 95%% has been achieved. ' ...
                    'Stopping training now.\n']);
                break;                                                      % Exits the loop if the accuracy is greater than or equal to 95%.
            end
        end
        if accuracy >= 0.95                                                 % Checks if the accuracy is greater than or equal to 95%.
            break;                                                          % Exits the loop if the accuracy is greater than or equal to 95%.
        end
    end
    if accuracy >= 0.95                                                     % Checks if the accuracy is greater than or equal to 95%.
        break;                                                              % Exits the loop if the accuracy is greater than or equal to 95%.
    end
end
% Prints messages in console about results acheived for better visualisation
fprintf('Best Optimal Parameters achieved :\n');                            % Prints a message indicating that the best optimal parameters have been achieved.
disp(bestParams);                                                           % Displays the best optimal parameters stored in the structure 'bestParams'.
fprintf('Best Average Accuracy obtained: %.2f%%\n', bestAccuracy * 100);    % Prints the best average accuracy obtained in a formatted string.
fprintf('Best Average Precision obtained: %.2f%%\n', bestPrecision * 100);  % Best average precision obtained in a formatted string.
fprintf('Time taken for best results: %.2f seconds\n', bestTrainingTime);   % Prints the time taken for achieving the best results in seconds in a formatted string.

% Bar Chart for Rice Grain Accuarcy
figure;                                                                     % Creates a new figure window for plotting
bar(1:numGrains, accuracyPerGrain);                                         % Creates a bar plot where each bar represents the accuracy for each unique grain label.
xticks(1:numGrains);                                                        % Sets the x-axis tick locations for the bar plot to be evenly spaced from 1 to 'numGrains'.
xticklabels(grainLabels);                                                   % Sets the x-axis tick labels for the bar plot to be the unique grain labels stored in the 'grainLabels' array.
xlabel('Grain Type');                                                       % X-Axis label named 'Grain Type'
ylabel('Accuracy');                                                         % Y-Axis label named 'Accuracy'
title('Accuracy for Different Types of Grains');                            % Adds a title to the plot indicating "Accuracy for Different Types of Grains".

% Bar Chart for Rice Grain Precision
figure;                                                                     % Creates a new figure window for plotting.
bar(1:numGrains, precisionPerGrain);                                        % Creates a bar plot where each bar represents the accuracy for each unique grain label.
xticks(1:numGrains);                                                        % Sets the x-axis tick locations for the bar plot to be evenly spaced from 1 to 'numGrains'.
xticklabels(grainLabels);                                                   % Sets the x-axis tick labels for the bar plot to be the unique grain labels stored in the 'grainLabels' array.
xlabel('Grain Type');                                                       % X-Axis label named 'Grain Type'
ylabel('Precision');                                                        % Y-Axis label named 'Accuracy'
title('Precision for Different Types of Grains');                           % This line adds a title to the plot indicating "Precision for Different Types of Grains".