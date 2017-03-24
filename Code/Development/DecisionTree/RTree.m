opts = detectImportOptions('Nashville_geocoded_processed.csv');
NashvilleT = readtable('Nashville_geocoded_processed.csv', opts);

%fill in missing data with an average. Find column averages first.
w = width(NashvilleT);
colaverages = zeros(1,w);
for k=2:w
    coldata = NashvilleT.(k);
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
    coldata = NashvilleT.(k);
    for n=1:length(coldata)
        if (coldata(n)==-1)
            NashvilleT.(k)(n) = colaverages(k);
        end
    end
end

%separate data into a training and testing set
x = height(NashvilleT);
TestSplit = floor(x*.9);
NashvilleTrain = NashvilleT(1:TestSplit,:);
NashvilleTest = NashvilleT(TestSplit+1:x,:);

%Sale_Price is the response variable
responsevar = NashvilleTrain.Sale_Price;
testresponse = NashvilleTest.Sale_Price;
NashvilleTrain.Sale_Price = [];
NashvilleTest.Sale_Price = [];

%Nashville Regression Tree Model with Cross Validation
RNTree = fitrtree(NashvilleTrain,responsevar,'Crossval', 'on');

%Use this line to find optimal parameters. I think this is contributing to
%overfitting though. Performance is worse when optimized.
%RNTreeOpt = fitrtree(NashvilleTrain,responsevar,'OptimizeHyperparameters','all');

predictions = predict(RNTree.Trained{1}, NashvilleTest);
%predictionsOpt = predict(RNTreeOpt, NashvilleTest);

%Calculate Mean Square Error for test set
MSE = loss(RNTree.Trained{1}, NashvilleTest, testresponse);
%MSEOpt = loss(RNTreeOpt, NashvilleTest, testresponse);

%view(RNTree.Trained{1},'Mode','graph');