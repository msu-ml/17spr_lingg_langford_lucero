source = sprintf('ART Dataset');
network = sprintf('regression');
adadelta = csvread('results_art_adadelta.csv', 1, 0);
adagrad = csvread('results_art_adagrad.csv', 1, 0);
sgd = csvread('results_art_sgd.csv', 1, 0);
n = 300;
clf();
figure(1);
plot(adadelta(1:n,1), adadelta(1:n,2), 'r--', ...
     adadelta(1:n,1), adadelta(1:n,4), 'r', ...
     adagrad(1:n,1), adagrad(1:n,2), 'g--', ...
     adagrad(1:n,1), adagrad(1:n,4), 'g', ...
     sgd(1:n,1), sgd(1:n,2), 'b--', ...
     sgd(1:n,1), sgd(1:n,4), 'b', ...
     'LineWidth', 1),
        set(gca, 'yscale', 'log');
ylim([0 1e-1])
title(source);
xlabel('Iteration');
ylabel('Loss');
legend('Adadelta (Training)', 'Adadelta (Test)', ...
       'Adagrad (Training)', 'Adagrad (Test)', ...
       'SGD (Training)', 'SGD (Test)', ...
       'Location', 'northeast');
%grid('on');