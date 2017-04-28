sets = ['Nashville_geocoded_processed.csv'; 'kc_house_data.csv               '; 'redfin_processed.csv            '];
datasets =  cellstr(sets);
datacount = length(datasets);
    
for h=1:datacount
    opts = detectImportOptions(datasets{h});
    DataT{h} = readtable(datasets{h}, opts);
    
    w = width(DataT{h});
    colaverages = zeros(1,w);
    for k=2:w
        coldata = DataT{h}.(k);
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
        coldata = DataT{h}.(k);
        for n=1:length(coldata)
            if (coldata(n)==-1)
                DataT{h}.(k)(n) = colaverages(k);
            end
        end
    end
end