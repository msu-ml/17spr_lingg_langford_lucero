opts = detectImportOptions('Nashville_geocoded_processed.csv');
NashvilleT = readtable('Nashville_geocoded_processed.csv', opts);

%make new classification, not normalizing because I think it already is
x = height(NashvilleT);
TestSplit = 5000;
NashvilleT.Sale_Class = zeros(x,1);

%Sort rows first
NashvilleTSort = sortrows(NashvilleT);

%create N equally sized classes
classCount = 8;
classSize = floor(x/classCount);
splits = zeros(classCount,1);

for t=1:classCount
    splits(t) = NashvilleTSort.(1)(t*classSize);
end

for i = 1:x
    for j = 1:classCount
        if (NashvilleT.Sale_Price(i) <= splits(j))
           NashvilleT.Sale_Class(i) = num2str(j);
           break
        end
    end
end

NashvilleT.Sale_Price = []; %removing sale price now that we have a sale class

%removing rows that have missing values for now (rows with a -1 value). 
%Will make a nearest neighbor replacement function later
W = width(NashvilleT);
for y = 1:W
    toDelete = NashvilleT.(y) == -1;
    NashvilleT(toDelete,:) = [];
end

x = height(NashvilleT);
NashvilleTrain = NashvilleT(1:TestSplit,:);
NashvilleTest = NashvilleT(TestSplit+1:x,:);

Response = NashvilleTrain.Sale_Class;

%Note, this is picky about column names. Strip out spaces, returns, 
%paranethesis, colons, and extra commas
CNTree = fitctree(NashvilleTrain, Response);

CNTree = prune(CNTree);
[score,cost] = predict(CNTree, NashvilleTest);
