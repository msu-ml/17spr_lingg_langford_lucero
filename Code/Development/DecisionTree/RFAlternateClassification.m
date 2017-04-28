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
    x = height(DataT);
    DataT.Sale_Class = zeros(x,1);
    DataTSort = sortrows(DataT, 'SALE_PRICE');
    
    classCount = 8;
    classSize = floor(x/classCount);
    splits = zeros(classCount,1);

    for t=1:classCount
        splits(t) = DataTSort.SALE_PRICE(t*classSize);
    end

    for i = 1:x
        for j = 1:classCount
            if (DataT.SALE_PRICE(i) <= splits(j))
               DataT.Sale_Class(i) = j;
               break
            end
        end
    end

    DataT.SALE_PRICE = []; %removing sale price now that we have a sale class

    
    %this will correspond to the number of features (subtract one for the
    %response column)
    w = width(DataT) - 1;
    
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
    TestResponse = DataTest.Sale_Class;
    DataTest.Sale_Class = [];
    XTest = table2array(DataTest);
    rng(0,'twister');
    for y=1:length(NumFeatures)
        Predictions = zeros(length(XTest),MaxTrees);
        for z=1:MaxTrees
            %Create a data replicate for every tree we are going to grow for
            %this ensemble. Datasample takes x random rows from the set with replacment. 
            DataReplicate = datasample(BootStrapData,x);

            %SALE_PRICE is the response variable
            Response = DataReplicate.Sale_Class;
            DataReplicate.Sale_Class = [];
            
            %We are using classregtree to grow each tree from a data replicate.
            %But first we have to convert the data table to a matrix, as
            %classregtree accepts a matrix as the X input. the 'nvartosample'
            %parameter will specify the number of features to select at random
            %to choose a data split from. This parameter is why matlab's newer
            %fitrtree function will not work, as it has no options for
            %specifying n var features to sample from. 
            X = table2array(DataReplicate);
            warning('off','all');
            %RFTree = TreeBagger(MaxTrees,BootStrapData,Response,'Method','classification','NumPredictorsToSample',NumFeatures(y));
            RFTree = classregtree(X, Response,'prune','off','nvartosample',NumFeatures(y),'method','classification');
            %RFTree = classregtree(X, Response,'prune','off','nvartosample',NumFeatures(y),'method','classification','minparent',ceil(x/20));
            intermediate = eval(RFTree,XTest);
            interMat = cell2mat(intermediate);
            Predictions(:,z) = str2num(interMat);
            %Use this line to find best features
            %FeatureImportance{d}{y,z} = varimportance(RFTree);
            
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
                %we want to take the mode
                AggregatePredictions(j) = mode(Predictions(j,z));
            end
        end
        
        %Calculate the error by taking the fraction of misclassifications
        %MSEs(d,y) = immse(AggregatePredictions, TestResponse);
        MSEs(d,y) = numel(find(AggregatePredictions~=TestResponse))/TestCount;
    end
end
finish = datestr(now)
save('Results500TreeAlternateClassifer')
%save('Results100TreeAlternateClassifer20FactorSplit')