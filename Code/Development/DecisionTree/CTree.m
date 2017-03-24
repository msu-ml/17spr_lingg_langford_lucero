%input files list. Input file s should have the sale price labelled as
%'Sale_Price'. All spaces and punctuation other than '_' should be removed.
%All categorical variables should be changed to binary. 
sets = ['Nashville_geocoded_processed.csv'; 'kc_house_data.csv               '; 'redfin_processed.csv            '];
datasets =  cellstr(sets);
datacount = length(datasets);
CNMSEs = cell(datacount);
CNPredictions = cell(datacount);
CNMSEsOpts = cell(datacount);
CNPredictionsOpts = cell(datacount);
CNTree = cell(datacount);
CNTreeOpt = cell(datacount);

for h=1:datacount
    opts = detectImportOptions(datasets{h});
    DataTable = readtable(datasets{h}, opts);

    %fill in missing data with an average. Find column averages first.
    w = width(DataTable);
    colaverages = zeros(1,w);
    for k=2:w
        coldata = DataTable.(k);
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
        coldata = DataTable.(k);
        for n=1:length(coldata)
            if (coldata(n)==-1)
                DataTable.(k)(n) = colaverages(k);
            end
        end
    end

    %make new classification, not normalizing because I think it already is
    x = height(DataTable);
    TestSplit = floor(x*.9);
    DataTable.Sale_Class = zeros(x,1);

    %Sort rows first
    DataTSort = sortrows(DataTable);

    %create N equally sized classes
    classCount = 8;
    classSize = floor(x/classCount);
    splits = zeros(classCount,1);

    for t=1:classCount
        splits(t) = DataTSort.(1)(t*classSize);
    end

    for i = 1:x
        for j = 1:classCount
            if (DataTable.Sale_Price(i) <= splits(j))
               DataTable.Sale_Class(i) = num2str(j);
               break
            end
        end
    end

    DataTable.Sale_Price = []; %removing sale price now that we have a sale class

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
    CNTreeOpt{h} = fitrtree(DataTrain,Response,'OptimizeHyperparameters','all');

    CNPredictions{h} = predict(CNTree{h}.Trained{1}, DataTest);
    CNPredictionsOpts{h} = predict(CNTreeOpt{h}, DataTest);
    
    CNMSEs{h} = loss(CNTree{h}.Trained{1}, DataTest, TestResponse);
    CNMSEsOpts{h} = loss(CNTreeOpt{h}, DataTest, TestResponse);

    %view(CNTree.Trained{1},'Mode','graph')
    
    DataTable = [];
    
    TestResponse = [];
    Response = [];

    DataTrain = [];
    DataTest = [];
end

CNMSE = cell2mat(CNMSEs);
CNMSEOpt = cell2mat(CNMSEsOpts);

plot(1:3, CNMSE, 'r--');
hold on
plot(1:3, CNMSEOpt, 'b--');
xlabel('Datasets');
ylabel('MSE');
legend('Classificication Crossfold MSE','Classification Optimized Crossfold MSE');
