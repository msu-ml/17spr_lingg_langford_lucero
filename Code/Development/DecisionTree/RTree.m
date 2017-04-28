%input files list. Input file s should have the sale price labelled as
%'Sale_Price'. All spaces and punctuation other than '_' should be removed.
%All categorical variables should be changed to binary. 
sets = ['nashville_processed.csv         '; 'kingcounty_processed.csv        '; 'redfin_processed.csv            '; 'art_processed.csv               '];
datasets =  cellstr(sets);
datacount = length(datasets);
NumTrees = [10, 50, 100];
NumParameters = 5;

RMSEs = cell(datacount);
RPredictions = cell(datacount);
RPredictionsOpts = cell(datacount);
RMSEOpts = cell(datacount);
TrainingRMSE = cell(datacount);
TrainingRMSEOpt = cell(datacount);
RNTree = cell(datacount);
RNTreeOpt = cell(datacount);

for h=1:datacount
    opts = detectImportOptions(datasets{h});
    DataT = readtable(datasets{h}, opts);

    %fill in missing data with an average. Find column averages first.
    w = width(DataT);
    colaverages = zeros(1,w);
    for k=2:w
        coldata = DataT.(k);
        colsum = 0;
        count = 0;
        for n=1:length(coldata)
            if (coldata(n)~=-1)
                colsum = colsum + coldata(n);
                count = count + 1;
            end
            colavg = colsum / count;
            colaverages(k) = colavg;
        end
    end
    %now replace with averages.
    for k=2:w
        coldata = DataT.(k);
        for n=1:length(coldata)
            if (coldata(n)==-1)
                DataT.(k)(n) = colaverages(k);
            end
        end
    end

    %separate data into a training and testing set
    x = height(DataT);
    TestSplit = floor(x*.9);
    DataTrain = DataT(1:TestSplit,:);
    DataTest = DataT(TestSplit+1:x,:);

    %Sale_Price is the response variable
    responsevar = DataTrain.SALE_PRICE;
    testresponse = DataTest.SALE_PRICE;
    DataTrain.SALE_PRICE = [];
    DataTest.SALE_PRICE = [];

    %Nashville Regression Tree Model with Cross Validation
    RNTree{h} = fitrtree(DataTrain,responsevar,'Crossval', 'on');
    
    %Use this line to find optimal parameters. I think this is contributing to
    %overfitting though. Performance is worse when optimized.
    RNTreeOpt{h} = fitrtree(DataTrain,responsevar,'OptimizeHyperparameters','all');

    RPredictions{h} = predict(RNTree{h}.Trained{1}, DataTest);
    RPredictionsOpts{h} = predict(RNTreeOpt{h}, DataTest);

    %Calculate Mean Square Error for test set
    RMSEs{h} = loss(RNTree{h}.Trained{1}, DataTest, testresponse);
    RMSEOpts{h} = loss(RNTreeOpt{h}, DataTest, testresponse);
    %Calculate MSE for Training set
    TrainingRMSE{h} = loss(RNTree{h}.Trained{1}, DataTrain, responsevar);
    TrainingRMSEOpt{h} = loss(RNTreeOpt{h}, DataTrain, responsevar);

    %view(RNTree.Trained{1},'Mode','graph');
    
    DataTable = [];
    
    TestResponse = [];
    Response = [];

    DataTrain = [];
    DataTest = [];

end
RMSEOpt = cell2mat(RMSEOpts);
RMSE = cell2mat(RMSEs);

%TrainRMSEOpt = cell2mat(TrainingRMSEOpts);
%TrainRMSE = cell2mat(TrainingRMSEs);
%{
plot(1:3, RMSE, 'r--');
hold on
plot(1:3, RMSEOpt, 'b--');
xlabel('Datasets');
ylabel('MSE');
legend('Test Regression Crossfold MSE', 'Test Regression Optimized MSE');
%}