tbl = readtable ( 'train.csv' );

tbl(1:5,:);

lm = fitlm ( tbl(1:size(tbl,1)/2,:), 'linear' );

figure;
lm.plot;

tblArray = table2array(tbl);
tblArray = sortrows(tblArray,size(tblArray,2));
disp('Run Error:');
disp(RunError(lm,tblArray(size(tbl,1)/2+1:size(tbl,1),size(tblArray,2)),tblArray(size(tbl,1)/2+1:size(tbl,1),1:size(tblArray,2)-1)));

  % Calculate the MSE where Data is the input Data,
  % Truth is the actual results corrisponding with the input Data,
  % ModelW and Bias are our model to test.
function MSE = RunError(Model, Truth, Data)
  MSE = 0;

    % Calculate the MSE of each data sample.
  result = zeros( length(Truth) );
  for i = 1:length(Truth)
    result( i ) = predict(Model,Data( i, : ));

    MSE = MSE + ( Truth( i ) - result ( i ) )^2;
  end
  
  figure;
  plot ( result, 'r' );
  hold on;
  plot ( Truth , 'g' );
  legend('Prediction','Truth');

    % Calculate the final MSE.
  MSE = 1/length(Truth) * MSE;
end
