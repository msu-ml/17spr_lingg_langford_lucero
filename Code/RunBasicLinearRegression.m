%nashErr = BasicLinearRegression('Processed/nashville_processed');
artErr = BasicLinearRegression('Processed/art_processed');
redErr = BasicLinearRegression('Processed/redfin_processed');
kingErr = BasicLinearRegression('Processed/kingcounty_processed');
%disp('Nashville:');
%disp(nashErr);
disp('ART:');
disp(artErr);
disp('Redfin:');
disp(redErr);
disp('Kings:');
disp(kingErr);