load DataWithAverages.mat;
DataT = AvgDataTables{1};
NumTrees = [50, 100, 100];
NumFeatures = [50];
MaxTrees = NumTrees(length(NumTrees));
MSE = zeros(length(NumFeatures),1);

%Create data replicates
%Forests{i} = zeros(NumTrees,1);

x = height(DataT);
ReplicateSplit = floor(x*.7);
DataReplicate = DataT(1:ReplicateSplit,:);
DataTest = DataT(ReplicateSplit+1:x,:);
TestResponse = DataTest.Sale_Price;
DataTest.Sale_Price = [];
rng(0,'twister');
for y=1:length(NumFeatures)
    for z=1:MaxTrees
        %duplicate table, then fill it in with randomly chosen rows from
        %first table
        %features = DataT.Properties.VariableNames;
        %get x data samples, with replacement, from only the 70% portion of the
        %table
        DataBags{z} = datasample(DataReplicate,x);

        %Sale_Price is the response variable
        ResponseVars{z} = DataBags{z}.Sale_Price;
        DataBags{z}.Sale_Price = [];
        %NumFeatures are the number of features we want in our bootstrap
        %replicate. So we generate a random permutation of feature indices to
        %remove with 1 to tablewidth as the range and tablewidth - NumFeatures
        %as the number of indices for removal to generate. This way there will
        %be NumFeatures left. 
        w = width(DataT) - 1;
        featuresToRemove = randperm(w-1, w-NumFeatures(y));
        DataBags{z}(:,featuresToRemove) = [];

        %Create tree model
        %RFTree{z} = fitrtree(DataBags{z},ResponseVars{z},'Crossval', 'off','Prune','off','MinLeafSize',500,'MaxNumSplits',30);
        RFTree{z} = fitrtree(DataBags{z},ResponseVars{z},'Crossval', 'off','Prune','off');
        %Make predictions
        RPredictions{z} = predict(RFTree{z}, DataTest);
        
        %alternate way to make trees
        %RFTree{z} = classregtree(table2array(DataBags{z}), ResponseVars{z},'prune','off','nvartosample',NumFeatures(y));

    end

    TestCount = height(DataTest);
    AggregatePredictions = zeros(TestCount,1);
    for z=1:MaxTrees
        for j=1:TestCount
            AggregatePredictions(j) = AggregatePredictions(j) + RPredictions{z}(j);
        end
    end
    for i=1:TestCount
        AggregatePredictions(i) = AggregatePredictions(i)/MaxTrees;
    end
    %Calculate error
    MSE(y) = immse(AggregatePredictions, TestResponse);

end