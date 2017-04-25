processedFileName = 'Processed/art_processed';
unprocessedNormFileName = 'Unprocessed/train_norm_price';

unprocessedErr = BasicLinearRegression(unprocessedNormFileName);
normErr = BasicLinearRegression(unprocessedNormFileName,true);
processedErr = BasicLinearRegression(processedFileName);
disp('Unprocessed File:');
disp(unprocessedErr);
disp('Unprocessed Nomalized File:');
disp(normErr);
disp('Processed File:');
disp(processedErr);
