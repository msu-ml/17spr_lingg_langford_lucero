fileName = 'train_processed';
tbl = readtable ( strcat('../../../../Data/Processed/',fileName,'.csv') );

tblArray = table2array(tbl);
%tblArray = tblArray(1:20000,:);
tblArray = cat(2,tblArray(:,1),normc(tblArray(:,2:size(tblArray,2))));

y_train = tblArray(1:size(tblArray,1)/2,1);
x_train = tblArray(1:size(tblArray,1)/2,2:size(tblArray,2));
y_test = tblArray(size(tblArray,1)/2 + 1:size(tblArray,1),1);
x_test = tblArray(size(tblArray,1)/2 + 1:size(tblArray,1),2:size(tblArray,2));

bestWeights = solveGradientDescent ( x_train , y_train );
disp(RunError(bestWeights,y_test,cat(2,ones(size(x_test,1),1),x_test),true,true,fileName));

tblArraySorted = sortrows(tblArray,1);
%tblArraySorted = cat(2,normc(tblArraySorted(:,1:size(tblArraySorted,2)-1)),tblArraySorted(:,size(tblArraySorted,2):size(tblArray,2)));

y_test_sorted = tblArraySorted(size(tblArray,1)/2 + 1:size(tblArray,1),1);
x_test_sorted = tblArraySorted(size(tblArray,1)/2 + 1:size(tblArray,1),2:size(tblArray,2));

disp('Run Error:');
disp(RunError(bestWeights,y_test_sorted,cat(2,ones(size(x_test_sorted,1),1),x_test_sorted),true,false));

function BestWeights = solveGradientDescent(TrainData, TrainSolution)
  BestWeights = zeros(1,size(TrainData,2) + 1);
  alpha = 1;
  dataWithBias = cat(2,ones(size(TrainData,1),1),TrainData);
  
  for a = 1:1000
  for i = 1:size(TrainSolution,1)
      error = BestWeights * dataWithBias( i, : )' - TrainSolution(i);
      BestWeights = BestWeights - alpha * error * dataWithBias( i, : );
  end
  disp(strcat(num2str(a),'%'));
  alpha = alpha * 0.99;
  end
    figure;
    plot ( BestWeights, 'r' );
    legend('BestWeights');
end

  % Calculate the MSE where Data is the input Data,
  % Truth is the actual results corrisponding with the input Data,
  % ModelW and Bias are our model to test.
function MSE = RunError(ModelW, Truth, Data, Graph, DumpData, FileName)
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

  if ( DumpData == true )
    save(strcat(FileName,'_lin_reg_predictions.mat'),'result');
  end
  
    % Calculate the final MSE.
  MSE = 1/length(Truth) * MSE;
end
