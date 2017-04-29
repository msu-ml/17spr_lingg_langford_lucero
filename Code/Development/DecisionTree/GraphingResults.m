load('AvgMSEForest.mat')
x = [1,2,3,4];
figure()
plot(x,AvgMSE50RTree,'r')
hold all
plot(x,AvgMSE100RTree,'g--')
plot(x,AvgMSE500RTree,'b--')
xticks([1 2 3 4])
xticklabels({'Nashville','KC','GR','Art'})
xlabel('Data Sets');
ylabel('Avg Mean Square Error');
legend('50 R Tree Forest','100 R Tree Forest','500 R Tree Forest');
title('Regression Forest Performance For Number of Trees');


figure()
plot(x,AvgMSE50CTree,'r')
hold all
plot(x,AvgMSE100CTree,'g--')
plot(x,AvgMSE500CTree,'b--')
xticks([1 2 3 4])
xticklabels({'Nashville','KC','GR','Art'})
xlabel('Data Sets');
ylabel('Avg Mean Square Error');
legend('50 C Tree Forest','100 C Tree Forest','500 C Tree Forest');
title('Classification Forest Performance For Number of Trees');