opts = detectImportOptions('Nashville_geocoded_processed.csv');
NashvilleT = readtable('Nashville_geocoded_processed.csv', opts);

%separate data into a training and testing set
x = height(NashvilleT);
TestSplit = 10000;
NashvilleTrain = NashvilleT(1:TestSplit,:);
NashvilleTest = NashvilleT(TestSplit+1:x,:);


%Sale_Price is the response variable
responsevar = NashvilleTrain.Sale_Price;
%Nashville Regression Tree Model, created using 5 KFold validation
NashvilleRTree = fitrtree(NashvilleTrain,responsevar,'KFold',5,'Prune','on',...
    'PruneCriterion','error','SplitCriterion','MSE');
%fitdata = NashvilleRTree.kfoldPredict(

[score,cost] = predict(NashvilleRTree, NashvilleTest);
