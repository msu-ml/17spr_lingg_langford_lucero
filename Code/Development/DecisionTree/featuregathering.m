%Run after one of the Feature Selection scripts or loading feature results.

sets = ['nashville_processed.csv         '; 'kingcounty_processed.csv        '; 'redfin_processed.csv            '; 'art_processed.csv               '];
datasets =  cellstr(sets);
datacount = length(datasets);

%Get all the feature names that will correspond to feature indices
for d=1:datacount
    dataset = datasets{d};
    opts = detectImportOptions(dataset);
    DataT = readtable(dataset, opts);
    DataT.SALE_PRICE = []; %response isn't counted as one of the features
    FeatureNames{d} = DataT.Properties.VariableNames;
    
end

datacount = 4; %number of datasets
MaxTrees = 100; %number of trees
featurevariance = 7; %number of feature selection options (length of NumFeatures)
NumberOfFeaturesToSelect = 10;

for d=1:datacount %represents each data set
    w = length(PredictorImportance{d,1});
    features = zeros(w,1);
    for y=1:featurevariance %for each feature set
        current = PredictorImportance{d,y}; %Each entry in 
            %PredictorImportance is an array that has a factor value for 
            %each feature that represents how important it is.
        
        for l=1:w %l represents the current feature
            features(l) = features(l) + current(l);
        end
    end
    
    for l=1:w %l represents the current feature
        features(l) = features(l) / featurevariance;
    end
    
    
    for i=1:NumberOfFeaturesToSelect
        index = 0; %index of the current greatest value
        f = 0; %the current greatest value
        for l=1:w %w is the number of features in this dataset
            if features(l) > f
                f = features(l); 
                index = l;
            end
        end
        features(index) = 0; %set the current highest to 0 in order 
        %to find the next highest
        if f > 0
            feats{i,1} = FeatureNames{d}{index};
            feats{i,2} = f;
        end
    end
    TopFeats{d} = feats;
end

%graph top features
for i=1:datacount
    figure();
    names = TopFeats{i}(:,1);
    for j=1:length(names)
        names(j) = strcat(int2str(j-1),names(j));
    end
    c = categorical(names);
    x = cell2mat(TopFeats{i}(:,2));
    
    bar(c,x);
end