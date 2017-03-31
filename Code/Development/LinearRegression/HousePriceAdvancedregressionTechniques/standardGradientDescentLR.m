tbl = readtable ( 'train.csv' );

tblArray = table2array(tbl);
tblArray = cat(2,normc(tblArray(:,1:size(tblArray,2)-1)),log(tblArray(:,size(tblArray,2):size(tblArray,2))));

tblArray = cat(2,tblArray(:,1:size(tblArray,2)-1),tblArray(:,1:size(tblArray,2)-1).^4,tblArray(:,size(tblArray,2)));

y_train = tblArray(1:size(tblArray,1)/2,size(tblArray,2));
x_train = tblArray(1:size(tblArray,1)/2,1:size(tblArray,2)-1);
y_test = tblArray(size(tblArray,1)/2 + 1:size(tblArray,1),size(tblArray,2));
x_test = tblArray(size(tblArray,1)/2 + 1:size(tblArray,1),1:size(tblArray,2)-1);

bestWeights = solveGradientDescent ( x_train , y_train );
disp(RunError(bestWeights,y_test,cat(2,ones(size(x_test,1),1),x_test),true));

tblArraySorted = sortrows(tblArray,size(tblArray,2));
tblArraySorted = cat(2,normc(tblArraySorted(:,1:size(tblArraySorted,2)-1)),tblArraySorted(:,size(tblArraySorted,2):size(tblArray,2)));

y_train_sorted = tblArraySorted(1:size(tblArray,1)/2,size(tblArray,2));
x_train_sorted = tblArraySorted(1:size(tblArray,1)/2,1:size(tblArray,2)-1);
y_test_sorted = tblArraySorted(size(tblArray,1)/2 + 1:size(tblArray,1),size(tblArray,2));
x_test_sorted = tblArraySorted(size(tblArray,1)/2 + 1:size(tblArray,1),1:size(tblArray,2)-1);

disp('Run Error:');
disp(RunError(bestWeights,y_train_sorted,cat(2,ones(size(x_train_sorted,1),1),x_train_sorted),true));
disp(RunError(bestWeights,y_test_sorted,cat(2,ones(size(x_test_sorted,1),1),x_test_sorted),true));

function BestWeights = solveGradientDescent(TrainData, TrainSolution)
  BestWeights = zeros(1,size(TrainData,2) + 1);
  alpha = 0.99;
  dataWithBias = cat(2,ones(size(TrainData,1),1),TrainData);
  
  for a = 1:1000
  for i = 1:size(TrainSolution,1)
    one = dataWithBias( i, : )' * dataWithBias( i, : ) * BestWeights';
    two = dataWithBias( i, : )' * TrainSolution( i );
    BestWeights = ( BestWeights' - alpha * ( one - two ) )';
  end
  alpha = alpha * 0.99;
  end
    figure;
    plot ( BestWeights, 'r' );
    legend('BestWeights');
end

  % Calculate the MSE where Data is the input Data,
  % Truth is the actual results corrisponding with the input Data,
  % ModelW and Bias are our model to test.
function MSE = RunError(ModelW, Truth, Data, Graph)
  MSE = 0;

    % Calculate the MSE of each data sample.
  result = zeros( length(Truth), 1 );
  for i = 1:length(Truth)
    result( i ) = ModelW * Data( i, : )';

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
