%Load the training data.
tbl = readtable ( 'train.csv' );

%Convert the table to an array, easier to work with.
tblArray = table2array(tbl);
%Normalize the data.
tblArray = cat(2,normc(tblArray(:,1:size(tblArray,2)-1)),tblArray(:,size(tblArray,2):size(tblArray,2)));

%Grab the truth value for later use.
truth = tblArray(:,size(tblArray,2));

%Convert house sale data to categories.
resultCategories = ones(size(tblArray,1),1);
for i = 1:size(tblArray,1)
   %represents 50k-750k in 10k increments
   for j = 1:150
    if tblArray(i,size(tblArray,2)) > ( j * 5000 + 50000 )
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
y_train = tblArray(1:size(tblArray,1)/2,size(tblArray,2));
x_train = tblArray(1:size(tblArray,1)/2,1:size(tblArray,2)-1);
resultCategories = resultCategories(1:size(tblArray,1)/2);

%Take the second half of the data for testing
y_test = tblArray(size(tblArray,1)/2 + 1:size(tblArray,1),size(tblArray,2));
x_test = tblArray(size(tblArray,1)/2 + 1:size(tblArray,1),1:size(tblArray,2)-1);

%Get the best logistic regression weights and test the model.
bestWeights = trainRanges ( x_train , resultCategories , categories );
disp(RunError(bestWeights,categories,y_test,cat(2,ones(size(x_test,1),1),x_test),true));

%Sort the data by sale price to produce an easier graph to look at.
tblArraySorted = sortrows(tblArray,size(tblArray,2));
tblArraySorted = cat(2,normc(tblArraySorted(:,1:size(tblArraySorted,2)-1)),tblArraySorted(:,size(tblArraySorted,2):size(tblArray,2)));

y_test_sorted = tblArraySorted(size(tblArray,1)/2 + 1:size(tblArray,1),size(tblArray,2));
x_test_sorted = tblArraySorted(size(tblArray,1)/2 + 1:size(tblArray,1),1:size(tblArray,2)-1);

disp('Run Error:');
disp(RunError(bestWeights,categories,y_test_sorted,cat(2,ones(size(x_test_sorted,1),1),x_test_sorted),true));

%Calculate the best weights from the training data matrix, the true output
%values for the training data and the number of categories.
%Note lower range is inclusive.
function BestWeights = trainRanges(TrainData, TrainSolution, NumCategories)
    %Initial weights just zeros, something closer to an initial
    %approximation may be better, close fit linear line?
  BestWeights = zeros(max(TrainSolution),size(TrainData,2) + 1);
  
  %Best step size found so far.
  alpha = 1.7;
  
  % Add the bias to the data as a column of 1s
  dataWithBias = cat(2,ones(size(TrainData,1),1),TrainData);
  
    %Perform 1000 training sweeps
  for a = 1:1000
      %Perform gradient descent on the model for each category.
    for j = 1:NumCategories
        %Loop through the entire data set and perform a single step of
        %gradient descent.
      for i = 1:size(TrainSolution,1)
        trainSolution = 0;
		
		  %Result is only 1 if this is a matching category.
        if TrainSolution(i) == j
          trainSolution = 1; 
        end
        error = 1 / (1 + exp( -BestWeights( j , : ) * dataWithBias( i, : )' ) ) - trainSolution;

        BestWeights( j , : ) = BestWeights( j , : ) - alpha * error * dataWithBias( i, : );
      end
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
    subResult = zeros(1, max(Truth));
    for j = 1:Categories
        subResult( j ) = 1 / (1 + exp( -ModelW(j,:) * Data( i, : )' ) );
    end
    [junk result( i )] = max(subResult);

    %Best category is multiplied by 5k and added with 50k to the value it
    %represents.
    MSE = MSE + ( Truth( i ) - ( result ( i ) * 5000 + 50000 ) )^2;
  end
  
  result = ( result * 5000 + 50000 );
  
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
