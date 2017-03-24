%input files list. Input file s should have the sale price labelled as
%'Sale_Price'. All spaces and punctuation other than '_' should be removed.
%All categorical variables should be changed to binary. 
sets = ['Nashville_geocoded_processed.csv'; 'kc_house_data.csv               '; 'redfin_processed.csv            '];
datasets =  cellstr(sets);
datacount = length(datasets);

RMSEs = cell(datacount);
RPredictions = cell(datacount);
RPredictionsOpts = cell(datacount);
RMSEOpts = cell(datacount);

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
    responsevar = DataTrain.Sale_Price;
    testresponse = DataTest.Sale_Price;
    DataTrain.Sale_Price = [];
    DataTest.Sale_Price = [];

    %Nashville Regression Tree Model with Cross Validation
    RNTree = fitrtree(DataTrain,responsevar,'Crossval', 'on');

    %Use this line to find optimal parameters. I think this is contributing to
    %overfitting though. Performance is worse when optimized.
    RNTreeOpt = fitrtree(DataTrain,responsevar,'OptimizeHyperparameters','all');

    RPredictions{h} = predict(RNTree.Trained{1}, DataTest);
    RPredictionsOpts{h} = predict(RNTreeOpt, DataTest);

    %Calculate Mean Square Error for test set
    RMSEs{h} = loss(RNTree.Trained{1}, DataTest, testresponse);
    RMSEOpts{h} = loss(RNTreeOpt, DataTest, testresponse);

    %view(RNTree.Trained{1},'Mode','graph');
    
    DataTable = [];
    
    TestResponse = [];
    Response = [];

    DataTrain = [];
    DataTest = [];

end

RMSEOpt = cell2mat(RMSEOpts);
RMSE = cell2mat(RMSEs);

plot(1:3, RMSE, 'r--');
hold on
plot(1:3, RMSEOpt, 'b--');
xlabel('Datasets');
ylabel('MSE');
legend('Regression Crossfold MSE', 'Regression Crossfold Optimized MSE');
