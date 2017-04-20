[nashvilleMSE, nashvilleAccuracy] = LogisticRegressionHeirarchy('nashville_processed', 100);
[redfinMSE, redfinAccuracy] = LogisticRegressionHeirarchy('redfin_processed', 100);
[artMSE, artAccuracy] = LogisticRegressionHeirarchy('art_processed', 100);
[kingMSE, kingAccuracy] = LogisticRegressionHeirarchy('kingcounty_processed', 100);

disp('Nashville Logistic Regression MSE:');
disp(nashvilleMSE);
disp(nashvilleAccuracy);
disp('Redfin Logistic Regression MSE:');
disp(redfinMSE);
disp(redfinAccuracy);
disp('ART Logistic Regression MSE:');
disp(artMSE);
disp(artAccuracy);
disp('King County MSE:');
disp(kingMSE);
disp(kingAccuracy);

function [MSE, Accuracy] = LogisticRegressionHeirarchy(FileName, Classes)
  %Load the training data.
  tbl = readtable ( strcat('../Data/Processed/New/',FileName,'.csv') );

  %Convert the table to an array, easier to work with.
  tblArray = table2array(tbl);

  %Grab the truth value for later use.
  truth = tblArray(:,size(tblArray,2));

  %Convert house sale data to categories.
  resultCategories = ones(size(tblArray,1),1);
  for i = 1:size(tblArray,1)
   for j = 1:Classes - 1
    if tblArray(i,size(tblArray,2)) > ( j * ( 1 / Classes ) )
        resultCategories( i ) = resultCategories( i ) + 1;
    else
        break;
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
  resultCategories_train = resultCategories(1:size(tblArray,1)/2);

  %Take the second half of the data for testing
  y_test = tblArray(size(tblArray,1)/2 + 1:size(tblArray,1),size(tblArray,2));
  x_test = tblArray(size(tblArray,1)/2 + 1:size(tblArray,1),1:size(tblArray,2)-1);
  resultCategories_test = resultCategories(size(tblArray,1)/2+1:size(tblArray,1));

  %Get the best logistic regression weights and test the model.
  bestWeights = trainRanges ( x_train , resultCategories_train , categories );
  [MSE, Accuracy] = RunError(bestWeights,categories,resultCategories_test./categories,y_test,cat(2,ones(size(x_test,1),1),x_test),true);
  disp(MSE);

  %Sort the data by sale price to produce an easier graph to look at.
  tblArraySorted = sortrows(tblArray,1);
  tblArraySorted = cat(2,tblArraySorted(:,size(tblArray,2)),normc(tblArraySorted(:,1:size(tblArraySorted,2)-1)));

  y_test_sorted = tblArraySorted(size(tblArray,1)/2 + 1:size(tblArray,1),size(tblArray,2));
  x_test_sorted = tblArraySorted(size(tblArray,1)/2 + 1:size(tblArray,1),1:size(tblArray,2)-1);

  disp('Run Error:');
  disp(RunError(bestWeights,categories,resultCategories_test./categories,y_test_sorted,cat(2,ones(size(x_test_sorted,1),1),x_test_sorted),true));
end

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
function [MSE, Accuracy] = RunError(ModelW, Categories, ResultCategories, Truth, Data, Graph)
  MSE = 0;

  accuratePredictions = 0;
  
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
    result( i ) = result( i ) * ( 1 / Categories );

    if ( abs ( result ( i ) - ResultCategories( i ) ) < ( 1 / Categories ) )
       accuratePredictions = accuratePredictions + 1; 
    end
	
    MSE = MSE + ( Truth( i ) - result ( i ) )^2;
  end
  
  if ( Graph == true )
    figure;
    plot ( result, 'r' );
    hold on;
    plot ( Truth , 'g' );
    legend('Prediction','Truth');
  end

  Accuracy = accuratePredictions / size( Truth , 1 );
  
    % Calculate the final MSE.
  MSE = 1/length(Truth) * MSE;
end
