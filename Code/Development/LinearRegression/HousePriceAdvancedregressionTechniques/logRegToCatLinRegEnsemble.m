fileName = 'train_processed';
tbl = readtable ( strcat('../../../../Data/Processed/',fileName,'.csv') );
load ( strcat('../../../../Data/Predictions/',fileName,'_log_reg_predictions_broad.mat') );

% Setup the lambda regularization values to test.
testLambdas( 1 ) = 1e-7;
testLambdas( 2 ) = 1.5e-7;
testLambdas( 3 ) = 1e-6;
testLambdas( 4 ) = 1.5e-6;
testLambdas( 5 ) = 1e-5;
testLambdas( 6 ) = 1e-4;
testLambdas( 7 ) = 1e-3;
testLambdas( 8 ) = 1e-2;
testLambdas( 9 ) = 1e-1;

tblArray = table2array(tbl);
% The first half is used for training the original logistic regression.
size1 = size(tblArray,1)/2+1;
size2 = size(tblArray,1);
tblArray = tblArray(size(tblArray,1)/2+1:size(tblArray,1),:);
categoryArray = uint8(result(1:size(result,2)/2) * 20);

% Train with the first half, truth values are the prediction minus the
% actual.
x_train = tblArray(1:size(tblArray,1)/2,2:size(tblArray,2));
y_train = tblArray(1:size(tblArray,1)/2,1);

% Test against the second half, truth is the house price.
x_test = tblArray(size(tblArray,1)/2+1:size(tblArray,1),2:size(tblArray,2));
y_test = tblArray(size(tblArray,1)/2+1:size(tblArray,1),1);

tblArraySorted = sortrows(tblArray,1);

bestLambdaError = 999999999999;
bestLambda = 0;

crossResults = zeros([1460 1]);

for categoryIndex = 1:max(categoryArray)

newArray = categoryArray';
catIndexes = find(newArray == categoryIndex);
x_category = x_train(catIndexes,:);
y_category = y_train(catIndexes,:);

if ( size(x_category,1) > 0 )

for lambdaIndex = 1:9
    %Get the cross validation results for the current lambda.
  crossResults( lambdaIndex ) = CrossValidation(testLambdas(lambdaIndex), y_category, x_category);
%  disp(strcat('Cross Result lambda = ',num2str(testLambdas( lambdaIndex )),':',num2str(crossResults( lambdaIndex ))));
  
  if ( bestLambdaError > crossResults( lambdaIndex ) )
     bestLambdaError = crossResults( lambdaIndex );
     bestLambda = testLambdas(lambdaIndex);
  end
end

%disp(strcat('Best Lambda: ',num2str(bestLambda)));

W_ML(:,categoryIndex) = ( bestLambda*eye(size(x_category,2)) + x_category'*x_category ) \ x_category' * y_category; 

end
end

    figure;
    plot ( W_ML, 'r' );
    legend('BestWeights');

disp(TestError(W_ML,categoryArray,y_test,x_test,true,true,fileName));

%x_test = tblArraySorted(size(tblArraySorted,1)/2+1,size(tblArraySorted,1),2:size(tblArraySorted,2));
%y_test = tblArraySorted(size(tblArraySorted,1)/2+1,size(tblArraySorted,1),1);

%disp('Run Error:');
%disp(RunError(W_ML,y_test,x_test,true,false));

  % Calculate the MSE where Data is the input Data,
  % Truth is the actual results corrisponding with the input Data,
  % ModelW and Bias are our model to test.
function MSE = CrossError(ModelW, Truth, Data, Graph, DumpData, FileName)
  MSE = 0;

    % Calculate the MSE of each data sample.
  result = zeros( 1, length(Truth) );
  for i = 1:length(Truth)
    result( i ) = ModelW' * Data( i, : )';

    MSE = MSE + ( Truth( i ) - result ( i ) )^2;
  end

  if ( Graph == true )
    figure;
    plot ( result, 'r' );
    hold;
    plot ( Truth, 'g' );
    legend('Prediction','Truth');
  end

  if ( DumpData == true )
    save(strcat(FileName,'_lin_reg_predictions.mat'),'result');
  end
  
    % Calculate the final MSE.
  MSE = 1/length(Truth) * MSE;
end

  % Perform 5 fold cross validation on the training data set for the given
  % lambda.
function MSE = CrossValidation(LambdaValue, Truth, Data)
  runningMSE = 0;

  y_train = Truth;
  x_train = Data;
  
    % Get the truncated number of entries in 1/5th
  crossLength = floor(length(y_train) / 5);

    % Perform each of the 4 cross validations.
  for i = 0:3
    startTrain = i * crossLength + 1;
    endTrain = i * crossLength + crossLength + 1;

    % Get the cross train data for the current 1/5th of the training set
    crossTrain = x_train(startTrain:endTrain,:);
    
      % Get the data to test our cross train against (the last 1/5th of the
      % training set).
    crossTestTruth = y_train(startTrain:endTrain);

    % Peform the current cross train.
    W_ML = ( LambdaValue*eye(size(crossTrain,2)) + crossTrain'*crossTrain ) \ crossTrain' * crossTestTruth;
    
      % Calculate the MSE of the model against the cross test data.
    runningMSE = runningMSE + CrossError(W_ML, y_train, x_train, false, false);
  end

    % Average the MSE across the 4 cross train runs.
  MSE = runningMSE / 4;
end

  % Calculate the MSE where Data is the input Data,
  % Truth is the actual results corrisponding with the input Data,
  % ModelW and Bias are our model to test.
function MSE = TestError(ModelW, Categories, Truth, Data, Graph, DumpData, FileName)
  MSE = 0;

    % Calculate the MSE of each data sample.
  result = zeros( 1, length(Truth) );
  for i = 1:length(Truth)
    result( i ) = ModelW(:,Categories(i))' * Data( i, : )';

    MSE = MSE + ( Truth( i ) - result ( i ) )^2;
  end

  if ( Graph == true )
    figure;
    plot ( result, 'r' );
    hold;
    plot ( Truth, 'g' );
    legend('Prediction','Truth');
  end

  if ( DumpData == true )
    save(strcat(FileName,'_lin_reg_predictions.mat'),'result');
  end
  
    % Calculate the final MSE.
  MSE = 1/length(Truth) * MSE;
end

