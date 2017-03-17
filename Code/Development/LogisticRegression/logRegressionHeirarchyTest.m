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
  alpha = 15.0;
  
  % Add the bias to the data as a column of 1s
  dataWithBias = cat(2,ones(size(TrainData,1),1),TrainData);
  
    %Perform training sweeps
  sweepCount = 500;
  for a = 1:sweepCount
      
        trainSolution = zeros(size(TrainSolution,1),max(TrainSolution));

            %Setup all of the training solutions.
            %A 1 indicates the house value is greater than or equal to this
            %category.
        for i = 1:size(TrainSolution,1)
        trainSolution(i,1:TrainSolution(i)) = 1;
        end
        
        error = 1 ./ (1 + exp( -dataWithBias * BestWeights' ) ) - trainSolution;

        BestWeights = BestWeights - alpha * error' * dataWithBias;

    if (mod(a,sweepCount/100)==0)
    disp(strcat(num2str(a/(sweepCount/100)),'%'));
    end
%    alpha = alpha * 0.999;
  end
  
  %Plot the resulting weights for debugging
  disp(size(BestWeights));
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
      
      %Each category indicates the house value is worth at least this
      %category's value.  Sum up all the positive results.
    subResult = zeros(1, max(Truth));
    for j = 1:Categories
        if ( 1 / (1 + exp( -ModelW(j,:) * Data( i, : )' ) ) > 0.5 )
           result( i ) = result( i ) + 1; 
        end
    end

    %Best category is multiplied by 5k and added with 50k to the value it
    %represents.
    result( i ) = result( i ) * 5000 + 50000;
	
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
