tbl = readtable ( 'train.csv' );

y_train = tblArray(1:size(tblArray,1)/2,size(tblArray,2));
x_train = tblArray(1:size(tblArray,1)/2,1:size(tblArray,2)-1);
y_test = tblArray(size(tblArray,1)/2 + 1:size(tblArray,1),size(tblArray,2));
x_test = tblArray(size(tblArray,1)/2 + 1:size(tblArray,1),1:size(tblArray,2)-1);

bestWeights = solveGradientDescent ( x_train , y_train );
disp(RunError(bestWeights,y_test,cat(2,ones(size(x_test,1),1),x_test),true));

y_test_sorted = tblArraySorted(size(tblArray,1)/2 + 1:size(tblArray,1),size(tblArray,2));
x_test_sorted = tblArraySorted(size(tblArray,1)/2 + 1:size(tblArray,1),1:size(tblArray,2)-1);

disp(RunError(bestWeights,y_test_sorted,cat(2,ones(size(x_test_sorted,1),1),x_test_sorted),true));

function BestWeights = solveGradientDescent(TrainData, TrainSolution)
  BestWeights = zeros(1,size(TrainData,2) + 1);
  alpha = 0.0000000003;
  
  for a = 1:20
  for i = 1:size(TrainSolution,1)
      dataWithBias = cat(2,1,TrainData( i, : ))
      error = BestWeights * dataWithBias' - TrainSolution(i)
      for j = 1:size(BestWeights,2)
          BestWeights( j ) = BestWeights( j ) - alpha * error * dataWithBias( j );
      end
  end
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
  result = zeros( length(Truth) );
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
