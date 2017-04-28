%input files list. Input file s should have the sale price labelled as
%'SALE_PRICE'. All spaces and punctuation other than '_' should be removed.
%All categorical variables should be changed to binary. 
sets = ['nashville_processed.csv         '; 'kingcounty_processed.csv        '; 'redfin_processed.csv            '; 'art_processed.csv               '];
datasets =  cellstr(sets);
datacount = length(datasets);
CNMSEs = cell(datacount);
TrainingCNMSEs = cell(datacount);
CNPredictions = cell(datacount);
CNMSEsOpts = cell(datacount);
TrainingCNMSEsOpts = cell(datacount);
CNPredictionsOpts = cell(datacount);
CNTree = cell(datacount);
CNTreeOpt = cell(datacount);

for h=1:datacount
    opts = detectImportOptions(datasets{h});
    DataTable = readtable(datasets{h}, opts);

    %fill in missing data with an average. Find column averages first.
    w = width(DataTable);
    
    %make new classification
    x = height(DataTable);
    TestSplit = floor(x*.9);
    DataTable.Sale_Class = zeros(x,1);

    %Sort rows first
    DataTSort = sortrows(DataTable,'SALE_PRICE');

    %create N equally sized classes
    classCount = 8;
    classSize = floor(x/classCount);
    splits = zeros(classCount,1);

    for t=1:classCount
        splits(t) = DataTSort.SALE_PRICE(t*classSize);
    end

    for i = 1:x
        for j = 1:classCount
            if (DataTable.SALE_PRICE(i) <= splits(j))
               DataTable.Sale_Class(i) = j;
               break
            end
        end
    end

    DataTable.SALE_PRICE = []; %removing sale price now that we have a sale class

    x = height(DataTable);
    DataTrain = DataTable(1:TestSplit,:);
    DataTest = DataTable(TestSplit+1:x,:);

    TestResponse = DataTest.Sale_Class;
    Response = DataTrain.Sale_Class;

    DataTrain.Sale_Class = [];
    DataTest.Sale_Class = [];

    %Note, this is picky about column names. Strip out spaces, returns, 
    %paranethesis, colons, and extra commas
    CNTree{h} = fitctree(DataTrain, Response, 'Crossval', 'on');
    CNTreeOpt{h} = fitctree(DataTrain,Response,'OptimizeHyperparameters','all');

    CNPredictions{h} = predict(CNTree{h}.Trained{1}, DataTest);
    CNPredictionsOpts{h} = predict(CNTreeOpt{h}, DataTest);
    
    CNMSEs{h} = loss(CNTree{h}.Trained{1}, DataTest, TestResponse);
    CNMSEsOpts{h} = loss(CNTreeOpt{h}, DataTest, TestResponse);
    
    TrainingCNMSEs{h} = loss(CNTree{h}.Trained{1}, DataTrain, Response);
    TrainingCNMSEsOpts{h} = loss(CNTreeOpt{h}, DataTrain, Response);

    %view(CNTree.Trained{1},'Mode','graph')
    
end

CNMSE = cell2mat(CNMSEs);
CNMSEOpt = cell2mat(CNMSEsOpts);
%{
plot(1:3, CNMSE, 'r--');
hold on
plot(1:3, CNMSEOpt, 'b--');
xlabel('Datasets');
ylabel('MSE');
legend('Testing Classification Crossfold MSE','Testing Classification Optimized MSE');
%}