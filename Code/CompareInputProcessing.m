processedFileName = 'Processed/train_processed_not_rand';
unprocessedNormFileName = 'Unprocessed/train_norm_price';

err = BasicLinearRegression(unprocessedNormFileName);
disp('Unprocessed File:');
disp(err);
err = BasicLinearRegression(unprocessedNormFileName,true);
disp('Unprocessed Nomalized File:');
disp(err);
err = BasicLinearRegression(processedFileName);
disp('Processed File:');
disp(err);
