%nashvilleMSE = LogisticRegressionHeirarchy('nashville_processed');
redfinMSE = LogisticRegressionHeirarchy('redfin_processed');
artMSE = LogisticRegressionHeirarchy('art_processed');
kingMSE = LogisticRegressionHeirarchy('kingcounty_processed');

%disp('Nashville Linear Ensemble MSE:');
%disp(nashvilleMSE);
disp('Redfin Linear Ensemble MSE:');
disp(redfinMSE);
disp('ART Linear Ensemble MSE:');
disp(artMSE);
disp('King County Linear Ensemble MSE:');
disp(kingMSE);

function MSE = LogisticRegressionHeirarchy(FileName)
  tbl = readtable ( strcat('../Data/Processed/',FileName,'.csv') );

  tblArray = table2array(tbl);

  data = tblArray(:,1:size(tblArray,2)-1);
  dataMean = mean(data,1);
  temp = ones(size(data,1),1)*dataMean;
  [U, E, V] = svd(data - ones(size(data,1),1)*dataMean);
  principals = (U * E);
  recon1 = principals(:,1:size(data,2)*0.25) * V(:,1:size(data,2)*0.25)' + ones(size(data,1),1)*dataMean;

  testArray = cat(2,tblArray(:,size(tblArray,2)),recon1(:,randperm(size(recon1,2))));
  %testArray = cat(2,tblArray(:,1),tblArray(:,2:size(tblArray,2)));
  testArraySorted = sortrows(testArray,1);

  y_train = testArray(1:size(testArray,1)/2,1);
  x_train = testArray(1:size(testArray,1)/2,2:size(testArray,2));
  y_weights = testArray(size(testArray,1)/2+1:size(testArray,1)*3/4,1);
  x_weights = testArray(size(testArray,1)/2+1:size(testArray,1)*3/4,2:size(testArray,2));
  y_test = testArray(size(testArray,1)*3/4 + 1:size(testArray,1),1);
  x_test = testArray(size(testArray,1)*3/4 + 1:size(testArray,1),2:size(testArray,2));

  ensembleSize = ceil(double(size(x_train,2)-1)/4.0);

  W_ML = zeros(4,ensembleSize);

  for i = 1:ensembleSize
    W_ML(:,i) = CalcWeights(y_train, x_train(:,(i-1)*4+1:min((i-1)*4+4,size(x_train,2))));
  end

  %testArray = cat(2,tblArray(:,1),tblArray(:,2:size(tblArray,2)));
  %testArraySorted = sortrows(testArray,1);

  %y_train = testArray(1:size(testArray,1)/2,1);
  %x_train = testArray(1:size(testArray,1)/2,2:size(testArray,2));
  %y_weights = testArray(size(testArray,1)/2+1:size(testArray,1)*3/4,1);
  %x_weights = testArray(size(testArray,1)/2+1:size(testArray,1)*3/4,2:size(testArray,2));
  %y_test = testArray(size(testArray,1)*3/4 + 1:size(testArray,1),1);
  %x_test = testArray(size(testArray,1)*3/4 + 1:size(testArray,1),2:size(testArray,2));

  Weights = TrainWeights(W_ML,y_weights,x_weights);
  %Weights = ones(ensembleSize,1)/ensembleSize;

  MSE = EnsembleError(W_ML,Weights,y_test,x_test,true,true,FileName);

  y_test = testArraySorted(size(testArraySorted,1)/2 + 1:size(testArraySorted,1),1);
  x_test = testArraySorted(size(testArraySorted,1)/2 + 1:size(testArraySorted,1),2:size(testArraySorted,2));

  disp('Run Error:');
  disp(EnsembleError(W_ML,Weights,y_test,x_test,true,false));
end

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

%    figure;
%    plot ( Truth, 'g' );
%    hold;
%    plot ( result, 'r' );
%    legend('Prediction','Truth');
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
    ClassifierWeights = inv( LambdaValue*eye(size(crossTrain,2)) + crossTrain'*crossTrain ) * crossTrain' * crossTestTruth;
    
      % Calculate the MSE of the model against the cross test data.
    crossTestData = x_train(4 * crossLength:size(x_train,1),:);
    testTruth = y_train(4 * crossLength:size(y_train,1));
    runningMSE = runningMSE + CrossError(ClassifierWeights, testTruth, crossTestData);
  end

    % Average the MSE across the 4 cross train runs.
  MSE = runningMSE / 4;
end

function ModelW = CalcWeights(Truth, Data)
  Data = cat(2,Data,zeros(size(Data,1),4-size(Data,2)));

  bestLambdaError = 999999999999;
  bestLambda = 0;

  testLambda = 1;
  testLambdaDir = 1;
  testLambdaDelta = 10;
  lastError = 0;
  currError = 999999999999999;

  while(abs(lastError - currError) > 0.0001)
    lastError = currError;

      %Get the cross validation results for the current lambda.
%    ModelW = inv( testLambda*eye(size(Data,2)) + Data'*Data ) * Data' * Truth; 
    currError = CrossValidation(testLambda, Truth, Data);
%    disp(strcat('Cross Result lambda = ',num2str(testLambda),':',num2str(currError)));
  
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

  %disp(strcat('Best Lambda: ',num2str(bestLambda)));

  ModelW = inv( bestLambda*eye(size(Data,2)) + Data'*Data ) * Data' * Truth; 
end

function ClassifierWeights = TrainWeights(ModelW, Truth, Data)
  ensembleSize = ceil(double(size(Data,2))/4.0);

  bestLambdaError = 999999999999;
  bestLambda = 0;

  testLambda = 1;
  testLambdaDir = 1;
  testLambdaDelta = 10;
  lastError = 0;
  currError = 999999999999999;
  
    ensembleData = zeros( size(Truth,1), ensembleSize );
    for i = 1:size(Truth,1)
      for j = 1:ensembleSize
        ensembleData( i, j ) = ModelW(1:min(4,size(Data,2)-((j-1)*4)),j)' * Data(i,(j-1)*4+1:min(size(Data,2),(j-1)*4+4))';
      end
    end
    
  while(abs(lastError - currError) > 0.000001)
    lastError = currError;
    currError = CrossValidation(testLambda,Truth,ensembleData);

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
  ClassifierWeights = inv( bestLambda*eye(size(ensembleData,2)) + ensembleData'*ensembleData ) * ensembleData' * Truth;
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

