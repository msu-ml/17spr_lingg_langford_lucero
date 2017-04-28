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
    
    rng(0,'twister');
    %SALE_PRICE is the response variable
    Response = DataT.Sale_Class;
    DataT.Sale_Class = [];
    for y=1:length(NumFeatures)
        %Create forest using Matlab's TreeBagger function, default behavior
        %samples with replacement
        %RF = TreeBagger(MaxTrees,BootStrapData,Response,'Method','classification','NumPredictorsToSample',NumFeatures(y));

        RF = TreeBagger(MaxTrees,DataT,Response,'Method','classification','NumPredictorsToSample',NumFeatures(y),'InBagFraction',.7,'OOBPrediction','on','OOBPredictorImportance','on');
        %RFTree{z} = classregtree(X, Response,'prune','off','nvartosample',NumFeatures(y));
        PredictorImportance{d,y} = RF.OOBPermutedPredictorDeltaError;
        
        %Calculate MSE of the prediction vs the actual
        %err = RF.oobError
        %MSEs(d,y) = err(MaxTrees);
        
        %For accuracy, if the difference between predicted value and actual is
        %within the normalized plus or minus 10k normalized value.


    end
end
finish = datestr(now)
save('Results500TreeMatlabClassificationFeatureImportance')