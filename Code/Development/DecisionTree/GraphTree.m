CNMSE = cell2mat(CNMSEs);
CNMSEOpt = cell2mat(CNMSEsOpts);

plot(1:3, CNMSE, 'r--');
hold on
plot(1:3, CNMSEOpt, 'b--');
xlabel('Datasets');
ylabel('MSE');
legend('Testing Classification Crossfold MSE','Testing Classification Optimized MSE');


%{
TRMSE = cell2mat(TrainingRMSE);
TRMSEOpt = cell2mat(TrainingRMSEOpt);
plot(1:3, TRMSE, 'r--');
hold on
plot(1:3, TRMSEOpt, 'b--');
xlabel('Datasets');
ylabel('MSE');
legend('Training Regression Crossfold MSE','Training Regression Optimized MSE');
%}



%{
RMSEOpt = cell2mat(RMSEOpts);
RMSE = cell2mat(RMSEs);

plot(1:3, RMSE, 'r--');
hold on
plot(1:3, RMSEOpt, 'b--');
xlabel('Datasets');
ylabel('MSE');
legend('Test Regression Crossfold MSE', 'Test Regression Optimized MSE');
%}

%{
TCNMSE = cell2mat(TrainingCNMSEs);
TCNMSEOpt = cell2mat(TrainingCNMSEsOpts);

plot(1:3, TCNMSE, 'r--');
hold on
plot(1:3, TCNMSEOpt, 'b--');
xlabel('Datasets');
ylabel('MSE');
legend('Training Classification Crossfold MSE','Training Classification Optimized MSE');
%}