processedFileName = 'Processed/train_processed_not_rand';
unprocessedNormFileName = 'Unprocessed/train_norm_price';

BasicLinearRegression(unprocessedNormFileName);
BasicLinearRegression(unprocessedNormFileName,true);
BasicLinearRegression(processedFileName);
