start = datestr(now)

sets = ['nashville_processed.csv         '; 'kingcounty_processed.csv        '; 'redfin_processed.csv            '; 'art_processed.csv               '];
datasets =  cellstr(sets);
datacount = length(datasets);

%each ensemble will be 500 trees
MaxTrees = 500;
%Toy Set: MaxTrees = 5; 

%Structure for storing all Mean Square Error results
MSEs = zeros(4,7);

for d=1:datacount
    %load the data
    dataset = datasets{d};
    opts = detectImportOptions(dataset);
    DataT = readtable(dataset, opts);
    
    %this will correspond to the number of features (subtract one for the
    %response column)
    w = width(DataT) - 1;
    x = height(DataT);
    %Create different ensembles sampling different numbers of features.
    %The range of number of features to try will be a precentage of the
    %number of features for each data set, since each data set has
    %different numbers of features.
    NumFeatures = [floor(sqrt(w)), ceil(w*.05), floor(w*.15), floor(w*.3), floor(w*.45), floor(w*.6), floor(w*.75)];
    %Toy Set: NumFeatures = [floor(w*.15)];
    
    %Next use 70 percent of the data for training and create bootstrap
    %replicates.
    ReplicateSplit = floor(x*.7);
    BootStrapData = DataT(1:ReplicateSplit,:);
    DataTest = DataT(ReplicateSplit+1:x,:);
    TestResponse = DataTest.SALE_PRICE;
    DataTest.SALE_PRICE = [];
    rng(0,'twister');
    %SALE_PRICE is the response variable
    Response = BootStrapData.SALE_PRICE;
    BootStrapData.SALE_PRICE = [];
    for y=1:length(NumFeatures)
        %Create forest using Matlab's TreeBagger function, default behavior
        %samples with replacement
        RF = TreeBagger(MaxTrees,BootStrapData,Response,'Method','regression','NumPredictorsToSample',NumFeatures(y));
        %RFTree{z} = classregtree(X, Response,'prune','off','nvartosample',NumFeatures(y));

        AggregatePredictions = predict(RF,DataTest);
        
        %Calculate MSE of the prediction vs the actual
        MSEs(d,y) = immse(AggregatePredictions, TestResponse);

    end
end
finish = datestr(now)