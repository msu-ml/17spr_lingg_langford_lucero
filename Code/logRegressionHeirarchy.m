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
  tbl = readtable ( strcat('../Data/Processed/',FileName,'.csv') );

  %Convert the table to an array, easier to work with.
  tblArray = table2array(tbl);

  %Grab the truth value for later use.
  truth = tblArray(:,size(tblArray,2));

  %Convert house sale data to categories.
  resultCategories = zeros(size(tblArray,1),1);
  for i = 1:size(tblArray,1)
   for j = 1:Classes - 1
    if tblArray(i,size(tblArray,2)) > ( j * ( 1 / Classes ) )
        resultCategories( i ) = resultCategories( i ) + 1;
    else
        break;
    end
   end
  end
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
  [MSE, Accuracy] = RunError(bestWeights,categories,resultCategories_test./categories,y_test,cat(2,ones(size(x_test,1),1),x_test),true,true,FileName);

  %Sort the data by sale price to produce an easier graph to look at.
  tblArraySorted = sortrows(tblArray,1);
  tblArraySorted = cat(2,tblArraySorted(:,1),normc(tblArraySorted(:,2:size(tblArraySorted,2))));

  y_test_sorted = tblArraySorted(size(tblArraySorted,1)/2 + 1:size(tblArraySorted,1),size(tblArray,2));
  x_test_sorted = tblArraySorted(size(tblArraySorted,1)/2 + 1:size(tblArraySorted,1),1:size(tblArraySorted,2)-1);

  disp(RunError(bestWeights,categories,resultCategories_test./categories,y_test_sorted,cat(2,ones(size(x_test_sorted,1),1),x_test_sorted),true,false));
end

%Calculate the best weights from the training data matrix, the true output
%values for the training data and the number of categories.
%Note lower range is inclusive.
function BestWeights = trainRanges(TrainData, TrainSolution, NumCategories)
    %Initial weights just zeros, something closer to an initial
    %approximation may be better, close fit linear line?
  BestWeights = zeros(NumCategories,size(TrainData,2) + 1);
  
  %Best step size found so far.
  alpha = 100.0;
  
  % Add the bias to the data as a column of 1s
  dataWithBias = cat(2,ones(size(TrainData,1),1),TrainData);
  
  trainSolution = zeros(size(TrainSolution,1),NumCategories);

            %Setup all of the training solutions.
            %A 1 indicates the house value is greater than or equal to this
            %category.
  for i = 1:size(TrainSolution,1)
    trainSolution(i,1:TrainSolution(i)) = 1;
  end
        
  previousError = 99999;
  
    %Perform training sweeps
  sweepCount = 1000;
  for a = 1:sweepCount
    error = 1 ./ (1 + exp( -dataWithBias * BestWeights' ) ) - trainSolution;

    BestWeights = BestWeights - alpha * error' * dataWithBias;
    
    normError = norm ( error, 2 );
    prevNormError = norm ( previousError, 2 );
    
    if ( norm ( error - previousError, 2 ) < 0.000001 )
        break;
    elseif ( norm ( error, 2 ) > norm ( previousError, 2 ) )
      alpha = alpha * 0.95;
    end;
    previousError = error;

    if (mod(a,sweepCount/100)==0)
      disp(strcat(num2str(a/(sweepCount/100)),'%'));
    end
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
function [MSE, Accuracy] = RunError(ModelW, Categories, ResultCategories, Truth, Data, Graph, DumpData, FileName)
  MSE = 0;

  accuratePredictions = 0;
  
    % Calculate the MSE of each data sample.
  result = zeros( 1, length(Truth) );
  for i = 1:length(Truth)
      
      %Each category indicates the house value is worth at least this
      %category's value.  Sum up all the positive results.
    subResult = zeros(1, Categories);
    for j = 1:Categories
        probability = 1 / (1 + exp( -ModelW(j,:) * Data( i, : )' ) );
        if ( probability > 0.5 )
           result( i ) = result( i ) + 1; 
        end
    end

    %Best category is multiplied by 5k and added with 35k to the value it
    %represents.
    result( i ) = result( i ) * ( 1 / Categories );
    
    if ( abs ( result ( i ) - ResultCategories( i ) ) < ( 1 / Categories ) )
       accuratePredictions = accuratePredictions + 1; 
    end
	
    MSE = MSE + ( Truth( i ) - result ( i ) )^2;
  end
  
  if ( DumpData == true )
    save(strcat(FileName,'_log_reg_predictions.mat'),'result');
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
