noneMSE = BasicLinearRegression('Processed/ReplacementFiles/redfin_processed_sub_none');
meanMSE = BasicLinearRegression('Processed/ReplacementFiles/redfin_processed_sub_mean');
closestMSE = BasicLinearRegression('Processed/ReplacementFiles/redfin_processed_sub_closest_value');
closestMeanMSE = BasicLinearRegression('Processed/ReplacementFiles/redfin_processed_sub_closest_mean');

display ( 'No substitution' );
display ( noneMSE );
display ( 'Mean substitution' );
display ( meanMSE );
display ( 'Closest substitution' );
display ( closestMSE );
display ( 'Closest Mean substitution' );
display ( closestMeanMSE );
