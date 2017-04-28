start = datestr(now)

SampleFeatureNumber = 7;

sets = ['nashville_processed.csv         '; 'kingcounty_processed.csv        '; 'redfin_processed.csv            '; 'art_processed.csv               '];
datasets =  cellstr(sets);
datacount = length(datasets);

%each ensemble will be X trees
MaxTrees = 3;
%Toy Set: MaxTrees = 5; 

%Structure for storing all Mean Square Error results
%MSEs = cell(datacount,1);
MSEs = zeros(datacount,SampleFeatureNumber);
%Structure for storing all Regression Tree Ensembles
RTrees = cell(datacount,1);



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

    %initialize the number of MSE values to store for this data set.
    %Each value in this array represents the MSE of one ensemble using a
    %specific number of random features for data replication
    %MSEs{d} = zeros(length(NumFeatures),1);
    
    %Next use 70 percent of the data for training and create bootstrap
    %replicates.
    ReplicateSplit = floor(x*.7);
    BootStrapData = DataT(1:ReplicateSplit,:);
    DataTest = DataT(ReplicateSplit+1:x,:);
    TestResponse = DataTest.SALE_PRICE;
    DataTest.SALE_PRICE = [];
    XTest = table2array(DataTest);
    rng(0,'twister');
	FeatureCount = zeros(w,d);
    for y=1:length(NumFeatures)
        Predictions = zeros(length(XTest),MaxTrees);
        for z=1:MaxTrees
            %Create a data replicate for every tree we are going to grow for
            %this ensemble. Datasample takes x random rows from the set with replacment. 
            DataReplicate = datasample(BootStrapData,x);

            %SALE_PRICE is the response variable
            Response = DataReplicate.SALE_PRICE;
            DataReplicate.SALE_PRICE = [];

            %We are using classregtree to grow each tree from a data replicate.
            %But first we have to convert the data table to a matrix, as
            %classregtree accepts a matrix as the X input. the 'nvartosample'
            %parameter will specify the number of features to select at random
            %to choose a data split from. This parameter is why matlab's newer
            %fitrtree function will not work, as it has no options for
            %specifying n var features to sample from. 
            X = table2array(DataReplicate);
            
            warning('off','all');
            %RFTree = classregtree(X, Response,'prune','off','nvartosample',NumFeatures(y),'minparent',ceil(x/20));
            RFTree = classregtree(X, Response,'prune','off','nvartosample',NumFeatures(y));
            %Create predictions using test data. 
			%Check what the first few feature splits are
			%add those to an array that contains counts of each feature split
			%FeatureCount(FeatureIndex,d) = FeatureCount(FeatureIndex) + 1;
            Predictions(:,z) = eval(RFTree,XTest);
            DataReplicate = [];
            FeatureImportance{d}{y,z} = varimportance(RFTree);
            %RFTree = [];
            %disp('Tree Number - Feature Number - Data Number')
            %disp(z)
            %disp(y)
            %disp(d)
            
        end
        %We now have a MaxTrees number of predictions based on Maxtree
        %number of different trees grown by selecting NumFeature[y] random
        %feature samples from MaxTree number of different data replicate
        %sets. 
        
        %Now average the predictions of all the trees
        TestCount = height(DataTest);
        AggregatePredictions = zeros(TestCount,1);
        for z=1:MaxTrees
            for j=1:TestCount
                AggregatePredictions(j) = AggregatePredictions(j) + Predictions(j,z);
            end
        end
        for i=1:TestCount
            AggregatePredictions(i) = AggregatePredictions(i)/MaxTrees;
        end
        %Calculate MSE of the prediction vs the actual
        MSEs(d,y) = immse(AggregatePredictions, TestResponse);

    end
end

finish = datestr(now)
%save('Results100TreeAlternateRegression')