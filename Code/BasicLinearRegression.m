function MSE = BasicLinearRegression(FileName,Normalize)
  tbl = readtable ( strcat('../Data/',FileName,'.csv') );

  tblArray = table2array(tbl);
  
    % Normalize the data if requested.
  if ( nargin > 1 )
    if ( Normalize == true )
      for i = 1:size(tblArray,2)
        tblArray(:,i) = (tblArray(:,i) )/(max(tblArray(:,i) ));
      end
    end
  end

    % Create sorted data for debug graph.
  tblArraySorted = sortrows(tblArray,1);

  bestLambdaError = 999999999999;
  bestLambda = 0;

  testLambda = 1;
  testLambdaDir = 1;
  testLambdaDelta = 10;
  lastError = 0;
  currError = 999999999999999;
  testIndex = 1;
  while(abs(lastError - currError) > 0.0000001)
    lastError = currError;
      
      %Get the cross validation results for the current lambda.
    currError = CrossValidation(testLambda, tblArray);
    disp(strcat('Cross Result lambda = ',num2str(testLambda),':',num2str(currError)));
    lambdaErrors(testIndex) = currError;
    lambdaVals(testIndex) = testLambda;
    testIndex = testIndex + 1;
  
      %Check if this lambda produces a better result.
    if ( lastError > currError )
      if ( bestLambdaError > currError )
         bestLambdaError = currError;
         bestLambda = testLambda;
       end
    else
      testLambdaDir = -testLambdaDir;
      testLambdaDelta = testLambdaDelta * 0.1;
    end
    testLambda = testLambda + testLambdaDir * testLambdaDelta;
  end

    %Print the best lambda for debug.
  disp(strcat('Best Lambda: ',num2str(bestLambda)));

    %Split the training and testing data, 50/50 or 90/10 split?
  y_train = tblArray(1:size(tblArray,1)/2,1);
  x_train = tblArray(1:size(tblArray,1)/2,2:size(tblArray,2));
  y_test = tblArray(size(tblArray,1)/2 + 1:size(tblArray,1),1);
  x_test = tblArray(size(tblArray,1)/2 + 1:size(tblArray,1),2:size(tblArray,2));

    %Train using the best lambda.
  W_ML = inv( bestLambda*eye(size(x_train,2)) + x_train'*x_train ) * x_train' * y_train; 

    %Print out the best weights for debug.
  figure;
  plot ( W_ML, 'r' );
  legend('BestWeights');

    %Calculate the MSE for the model.
  if mean(y_test) > 1
    predictions = CalculatePredictions(W_ML,x_test,true,FileName);
    MSE = RunError(y_test / max(y_test), predictions / max(predictions),true);
  else
    MSE = RunError(y_test, CalculatePredictions(W_ML,x_test,true,FileName),true);
  end

    %Calculate the error of the sorted test data and display it for debug.
    %This approach is easier to eyeball.
  y_test = tblArraySorted(size(tblArray,1)/2 + 1:size(tblArray,1),1);
  x_test = tblArraySorted(size(tblArray,1)/2 + 1:size(tblArray,1),2:size(tblArray,2));
  disp(RunError(y_test, CalculatePredictions(W_ML,x_test, false),false));
end

  %Calculate the predictions vector from the model and input data.
  %Also dump the results to be used as an ensbemble input if requested.
function Predictions = CalculatePredictions(ModelW, Data, DumpData, FileName)
    Predictions = ModelW' * Data';
    Predictions = Predictions;
    if ( DumpData == true )
      result = Predictions;

      outputFileName = split(FileName,'/');
      fileName = char(strcat(outputFileName(2),'_lin_reg_predictions.mat'));
      save(fileName,'result');
    end
end

  % Calculate the MSE between the truth and predictions.
  % Graph the two if requested.
function MSE = RunError(Truth, Predictions, Graph)
  MSE = 0;

    % Calculate the MSE of each data sample.  Matlab probably has a way to
    % do this without iterating...
  for i = 1:length(Truth)
    MSE = MSE + ( Truth( i ) - Predictions ( i ) )^2;
  end

    %Graph if requested.
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
    runningMSE = runningMSE + RunError(testTruth, CalculatePredictions(W_ML, crossTestData, false),false);
  end

    % Average the MSE across the 4 cross train runs.
  MSE = runningMSE / 4;
end