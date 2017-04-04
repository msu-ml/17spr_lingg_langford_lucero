fileName = 'train_processed';
tbl = readtable ( strcat('../../../../Data/Processed/',fileName,'.csv') );

tblArray = table2array(tbl);
tblArray = cat(2,tblArray(:,1),tblArray(:,randperm(size(tblArray,2)-1)+1));
tblArraySorted = sortrows(tblArray,1);

y_train = tblArray(1:size(tblArray,1)/2,1);
x_train = tblArray(1:size(tblArray,1)/2,2:size(tblArray,2));
y_weights = tblArray(size(tblArray,1)/2+1:size(tblArray,1)*3/4,1);
x_weights = tblArray(size(tblArray,1)/2+1:size(tblArray,1)*3/4,2:size(tblArray,2));
y_test = tblArray(size(tblArray,1)*3/4 + 1:size(tblArray,1),1);
x_test = tblArray(size(tblArray,1)*3/4 + 1:size(tblArray,1),2:size(tblArray,2));

ensembleSize = ceil(double(size(tblArray,2)-1)/4.0);

W_ML = zeros(4,ensembleSize);

for i = 1:ensembleSize
  W_ML(:,i) = CalcWeights(y_train, x_train(:,(i-1)*4+1:min(size(x_train,2),(i-1)*4+4)));
end

Weights = TrainWeights(W_ML,y_weights,x_weights);

disp(EnsembleError(W_ML,Weights,y_test,x_test,true,true,fileName));

y_test = tblArraySorted(size(tblArray,1)/2 + 1:size(tblArray,1),1);
x_test = tblArraySorted(size(tblArray,1)/2 + 1:size(tblArray,1),2:size(tblArray,2));

disp('Run Error:');
disp(EnsembleError(W_ML,Weights,y_test,x_test,true,false));

  % Calculate the MSE where Data is the input Data,
  % Truth is the actual results corrisponding with the input Data,
  % ModelW and Bias are our model to test.
function MSE = CrossError(ModelW, Truth, Data)
  MSE = 0;

    % Calculate the MSE of each data sample.
  result = zeros( 1, size(Truth,1) );
  for i = 1:size(Truth,1)
    result( i ) = ModelW(:)' * Data(i,:)';

    MSE = MSE + ( Truth( i ) - result ( i ) )^2;
  end

    % Calculate the final MSE.
  MSE = 1/length(Truth) * MSE;
end

  % Perform 5 fold cross validation on the training data set for the given
  % lambda.
function MSE = CrossValidation(LambdaValue, ModelW, Truth, Data)
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
    ClassifierWeights = ( LambdaValue*eye(size(crossTrain,2)) + crossTrain'*crossTrain ) \ crossTrain' * crossTestTruth;
    
      % Calculate the MSE of the model against the cross test data.
    crossTestData = x_train(4 * crossLength:length(x_train),:);
    testTruth = y_train(4 * crossLength:length(y_train));
    runningMSE = runningMSE + CrossError(ClassifierWeights, testTruth, crossTestData);
  end

    % Average the MSE across the 4 cross train runs.
  MSE = runningMSE / 4;
end

function ModelW = CalcWeights(Truth, Data)
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

  bestLambdaError = 999999999999;
  bestLambda = 1;

  for lambdaIndex = 1:9
      %Get the cross validation results for the current lambda.
    ModelW = ( lambdaIndex*eye(size(Data,2)) + Data'*Data ) \ Data' * Truth; 
    crossResults = CrossError(ModelW, Truth, Data);
  
    if ( bestLambdaError > crossResults )
       bestLambdaError = crossResults;
       bestLambda = testLambdas(lambdaIndex);
    end
  end

  %disp(strcat('Best Lambda: ',num2str(bestLambda)));

  ModelW = ( bestLambda*eye(size(Data,2)) + Data'*Data ) \ Data' * Truth; 
end

function ClassifierWeights = TrainWeights(ModelW, Truth, Data)
  ensembleSize = ceil(double(size(Data,2))/4.0);

  testLambdas( 1 ) = 1e-7;
  testLambdas( 2 ) = 1.5e-7;
  testLambdas( 3 ) = 1e-6;
  testLambdas( 4 ) = 1.5e-6;
  testLambdas( 5 ) = 1e-5;
  testLambdas( 6 ) = 1e-4;
  testLambdas( 7 ) = 1e-3;
  testLambdas( 8 ) = 1e-2;
  testLambdas( 9 ) = 1e-1;

  bestLambdaError = 999999999999;
  bestLambda = 1;
  
    ensembleData = zeros( size(Truth,1), ensembleSize );
    for i = 1:size(Truth,1)
      for j = 1:ensembleSize
        ensembleData( i, j ) = ModelW(1:min(4,size(Data,2)-((j-1)*4)),j)' * Data(i,(j-1)*4+1:min(size(Data,2),(j-1)*4+4))';
      end
    end
    
  for lambdaIndex = 1:9
    crossResults = CrossValidation(testLambdas(lambdaIndex),ModelW,Truth,ensembleData);
    if ( bestLambdaError > crossResults )
       bestLambdaError = crossResults;
       bestLambda = testLambdas(lambdaIndex);
    end
  end
  ClassifierWeights = ( bestLambda*eye(size(ensembleData,2)) + ensembleData'*ensembleData ) \ ensembleData' * Truth;
end

  % Calculate the MSE where Data is the input Data,
  % Truth is the actual results corrisponding with the input Data,
  % ModelW and Bias are our model to test.
function MSE = EnsembleError(ModelW, ClassifierWeights, Truth, Data, Graph, DumpData, FileName)
  MSE = 0;
  ensembleSize = ceil(double(size(Data,2))/4.0);

    % Calculate the MSE of each data sample.
  ensembleData = zeros( ensembleSize, 1 );
  result = zeros( 1, size(Truth,1) );
  for i = 1:size(Data,1)
    for j = 1:ensembleSize
      ensembleData( j ) = ModelW(1:min(4,size(Data,2)-((j-1)*4)),j)' * Data(i,(j-1)*4+1:min(size(Data,2),(j-1)*4+4))';
    end
    result( i ) = ClassifierWeights' * ensembleData;
    MSE = MSE + ( Truth( i ) - result ( i ) )^2;
  end

  if ( Graph == true )
    figure;
    plot ( Truth, 'g' );
    hold;
    plot ( result, 'r' );
    legend('Prediction','Truth');
  end

  if ( DumpData == true )
    save(strcat(FileName,'_lin_reg_predictions.mat'),'result');
  end
  
    % Calculate the final MSE.
  MSE = 1/length(Truth) * MSE;
end
