function MSE = BasicLinearRegression(FileName,Normalize)
  tbl = readtable ( strcat('../Data/',FileName,'.csv') );

  % Setup the lambda regularization values to test.
  testLambdas( 1 ) = 1e-7;
  testLambdas( 2 ) = 1.5e-7;
  testLambdas( 3 ) = 1e-6;
  testLambdas( 4 ) = 1.5e-6;
  testLambdas( 5 ) = 1e-5;
  testLambdas( 6 ) = 1.5e-5;
  testLambdas( 7 ) = 1e-4;
  testLambdas( 8 ) = 1.5e-4;
  testLambdas( 9 ) = 1e-3;
  testLambdas( 10 ) = 1.5e-3;
  testLambdas( 11 ) = 1e-2;
  testLambdas( 12 ) = 1.5e-2;
  testLambdas( 13 ) = 1e-1;

  tblArray = table2array(tbl);
  
  if ( nargin > 1 )
    if ( Normalize == true )
      for i = 1:size(tblArray,2)
        tblArray(:,i) = (tblArray(:,i) - min(tblArray(:,i)))/(max(tblArray(:,i) - min(tblArray(:,i))));
      end
    end
  end

  tblArraySorted = sortrows(tblArray,1);

  bestLambdaError = 999999999999;
  bestLambda = 0;

  crossResults = zeros([1460 1]);

  for lambdaIndex = 1:size(testLambdas,1)
      %Get the cross validation results for the current lambda.
    crossResults( lambdaIndex ) = CrossValidation(testLambdas(lambdaIndex), tblArray);
    disp(strcat('Cross Result lambda = ',num2str(testLambdas( lambdaIndex )),':',num2str(crossResults( lambdaIndex ))));
  
    if ( bestLambdaError > crossResults( lambdaIndex ) )
       bestLambdaError = crossResults( lambdaIndex );
       bestLambda = testLambdas(lambdaIndex);
    end
  end

  disp(strcat('Best Lambda: ',num2str(bestLambda)));

  y_train = tblArray(1:size(tblArray,1)/2,1);
  x_train = tblArray(1:size(tblArray,1)/2,2:size(tblArray,2));
  y_test = tblArray(size(tblArray,1)/2 + 1:size(tblArray,1),1);
  x_test = tblArray(size(tblArray,1)/2 + 1:size(tblArray,1),2:size(tblArray,2));

  W_ML = inv( bestLambda*eye(size(x_train,2)) + x_train'*x_train ) * x_train' * y_train; 

    figure;
    plot ( W_ML, 'r' );
    legend('BestWeights');

    MSE = RunError(y_test, CalculatePredictions(W_ML,y_test,x_test,true,FileName),true);

    %Calculate the error of the sorted test data and display it for debug
  y_test = tblArraySorted(size(tblArray,1)/2 + 1:size(tblArray,1),1);
  x_test = tblArraySorted(size(tblArray,1)/2 + 1:size(tblArray,1),2:size(tblArray,2));
  disp(RunError(y_test, CalculatePredictions(W_ML,y_test,x_test, false),false));
end

function Predictions = CalculatePredictions(ModelW, Truth, Data, DumpData, FileName)
    Predictions = ModelW' * Data';
    if ( DumpData == true )
      result = Predictions;

      outputFileName = split(FileName,'/');
      fileName = char(strcat(outputFileName(2),'_lin_reg_predictions.mat'));
      save(fileName,'result');
    end
end
  % Calculate the MSE where Data is the input Data,
  % Truth is the actual results corrisponding with the input Data,
  % ModelW and Bias are our model to test.
function MSE = RunError(Truth, Predictions, Graph)
  MSE = 0;

    % Calculate the MSE of each data sample.
  for i = 1:length(Truth)
    MSE = MSE + ( Truth( i ) - Predictions ( i ) )^2;
  end

  if ( Graph == true )
    figure;
    plot ( Predictions, 'r' );
    hold;
    plot ( Truth, 'g' );
    legend('Prediction','Truth');
  end

    % Calculate the final MSE.
  MSE = 1/length(Truth) * MSE;
end

  % Perform 5 fold cross validation on the training data set for the given
  % lambda.
function MSE = CrossValidation(LambdaValue, TrainData)
  runningMSE = 0;

  y_train = TrainData(:,1);
  x_train = TrainData(:,2:size(TrainData,2));  
  
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
    W_ML = inv( LambdaValue*eye(size(crossTrain,2)) + crossTrain'*crossTrain ) * crossTrain' * crossTestTruth;
    
      % Calculate the MSE of the model against the cross test data.
    crossTestData = x_train(4 * crossLength:length(x_train),:);
    testTruth = y_train(4 * crossLength:length(y_train));
    runningMSE = runningMSE + RunError(testTruth, CalculatePredictions(W_ML, testTruth, crossTestData, false),false);
  end

    % Average the MSE across the 4 cross train runs.
  MSE = runningMSE / 4;
end