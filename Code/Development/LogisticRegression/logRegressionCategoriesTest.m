%Load the training data.
fileName = 'train_processed';
tbl = readtable ( strcat('../../../Data/Processed/',fileName,'.csv') );

%Convert the table to an array, easier to work with.
tblArray = table2array(tbl);

%Grab the truth value for later use.
truth = tblArray(:,size(tblArray,2));

%Convert house sale data to categories.
resultCategories = ones(size(tblArray,1),1);
for i = 1:size(tblArray,1)
   for j = 1:99
       
    if tblArray(i,1) > ( j * 0.01 )
        resultCategories( i ) = resultCategories( i ) + 1;
    end
   end
end
%tblArray(:,size(tblArray,2)) = resultCategories;
categories = max(resultCategories);

%Display the new categories for debug.
    figure;
    plot ( resultCategories, 'r' );
    legend('Truth');

%Take half the data for training
y_train = tblArray(1:size(tblArray,1)/2,1);
x_train = tblArray(1:size(tblArray,1)/2,2:size(tblArray,2));
resultCategories = resultCategories(1:size(tblArray,1)/2);

%Take the second half of the data for testing
y_test = tblArray(size(tblArray,1)/2 + 1:size(tblArray,1),1);
x_test = tblArray(size(tblArray,1)/2 + 1:size(tblArray,1),2:size(tblArray,2));

%Get the best logistic regression weights and test the model.
bestWeights = trainRanges ( x_train , resultCategories , categories );
disp(RunError(bestWeights,categories,y_test,cat(2,ones(size(x_test,1),1),x_test),true));

%Sort the data by sale price to produce an easier graph to look at.
tblArraySorted = sortrows(tblArray,1);
tblArraySorted = cat(2,tblArraySorted(:,1),normc(tblArraySorted(:,2:size(tblArraySorted,2))));

y_test_sorted = tblArraySorted(size(tblArray,1)/2 + 1:size(tblArray,1),1);
x_test_sorted = tblArraySorted(size(tblArray,1)/2 + 1:size(tblArray,1),2:size(tblArray,2));

disp('Run Error:');
disp(RunError(bestWeights,categories,y_test_sorted,cat(2,ones(size(x_test_sorted,1),1),x_test_sorted),true));

%Calculate the best weights from the training data matrix, the true output
%values for the training data and the number of categories.
%Note lower range is inclusive.
function BestWeights = trainRanges(TrainData, TrainSolution, NumCategories)
    %Initial weights just zeros, something closer to an initial
    %approximation may be better, close fit linear line?
  BestWeights = zeros(NumCategories,size(TrainData,2) + 1);
  
  alpha = 10.0;
  
  % Add the bias to the data as a column of 1s
  dataWithBias = cat(2,ones(size(TrainData,1),1),TrainData);
  
  previousError = 99999;
  
    %Perform 1000 training sweeps
  sweepCount = 1000;
  for a = 1:sweepCount
      totalError = 0;
      
        %Loop through the entire data set and perform a single step of
        %gradient descent.
      for i = 1:size(TrainSolution,1)
        trainSolution = zeros(1,NumCategories);

          %Result is 1 if the house value is greater than or equal to this category.
        trainSolution(1,TrainSolution(i)) = 1;
        
        error = 1 ./ (1 + exp( -dataWithBias( i, : ) * BestWeights' ) ) - trainSolution;
        BestWeights = BestWeights - alpha * error' * dataWithBias( i, : );
        
        totalError = totalError + norm( error, 2 );
      end
      
    if ( abs ( totalError - previousError ) < 0.00001 )
        break;
    elseif ( abs ( totalError ) > abs ( previousError ) )
      alpha = alpha * 0.5;
    end;
    previousError = totalError;
    
    if (mod(a,sweepCount/100)==0)
    disp(strcat(num2str(a/(sweepCount/100)),'%'));
    end
  end
  
  %Plot the resulting weights for debugging
    figure;
    plot ( BestWeights, 'r' );
    legend('BestWeights');
end

  % Calculate the MSE where Data is the input Data,
  % Truth is the actual results corrisponding with the input Data,
  % ModelW and Bias are our model to test.
function MSE = RunError(ModelW, Categories, Truth, Data, Graph)
  MSE = 0;

    % Calculate the MSE of each data sample.
  result = zeros( 1, length(Truth) );
  for i = 1:length(Truth)
      
      %Calculate the logistic regression for each category and take the
      %highest percentage result.
    subResult = zeros(1, Categories);
    for j = 1:Categories
        subResult( j ) = 1 / (1 + exp( -ModelW(j,:) * Data( i, : )' ) );

    end
    [junk result( i )] = max(subResult);
    result( i ) = result( i ) / 100.0;

    MSE = MSE + ( Truth( i ) - result ( i ) )^2;
  end
  
  if ( Graph == true )
    figure;
    plot ( result, 'r' );
    hold on;
    plot ( Truth , 'g' );
    legend('Prediction','Truth');
  end

    % Calculate the final MSE.
  MSE = 1/length(Truth) * MSE;
end
