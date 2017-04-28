RegressionErrors = [SingleRTreeMSEs(1), SingleRTreeMSEs(2), SingleRTreeMSEs(3), SingleRTreeMSEs(4); ...
SingleRTreeMSEsOptimized(1), SingleRTreeMSEsOptimized(2), SingleRTreeMSEsOptimized(3), SingleRTreeMSEsOptimized(4); ...
MatlabRegressionBestMSEs(1), MatlabRegressionBestMSEs(2), MatlabRegressionBestMSEs(3), MatlabRegressionBestMSEs(4); ...
MyRegressionBestMSEs(1), MyRegressionBestMSEs(2), MyRegressionBestMSEs(3), MyRegressionBestMSEs(4)];

r = categorical({'Single Regression Tree','Single Regression Tree Optimized', 'Random Forest Regression Matlab', 'Random Forest Regression Custom'});
figure()
bar(r,RegressionErrors);
ylabel('Mean Square Error');


ClassificationErrors = zeros(4,4);
for i=1:4
    ClassificationErrors(1,i) = SingleCTreeMSEs(i);
end
for i=1:4
    ClassificationErrors(2,i) = SingleCTreeMSEsOptimized(i);
end
for i=1:4
    ClassificationErrors(3,i) = MatlabClassificationBestMSEs(i);
end
for i=1:4
    ClassificationErrors(4,i) = MyClassificationBestMSEs(i);
end

c = categorical({'Single Classifier Tree','Single Classifier Tree Optimized', 'Random Forest Classifier Matlab', 'Random Forest Classifier Custom'});
figure()
bar(c,ClassificationErrors);
ylabel('Misclassification Rate');