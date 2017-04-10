classdef Experiment < handle
    properties
        sources
    end
    methods
        function obj = Experiment()
            fprintf('\nLoading data.\n');
            obj.sources = { ...
                %HousingData('Data/nashville_processed.csv', 'Nashville'), ...
                HousingData('Data/kingcounty_processed.csv', 'KingCounty'), ...
                HousingData('Data/redfin_processed.csv', 'GrandRapids'), ...
                HousingData('Data/art_processed.csv', 'ART') ...
                          };
        end
        function run(obj)
            n_sources = length(obj.sources);
            for i = 1:n_sources
                data = obj.sources{i};
                    fprintf('\nData ----------------------------------\n');
                    [data_train, data_test] = data.split_data(2, 1);
                    obj.display_data(data, data_train, data_test);
                    
                    fprintf('\nModel ---------------------------------\n');
                    layers = [data.num_features 35 15 10 1];
                    network = RegressNet(layers);
                    y_min = data.unnormalize_target(0.0);
                    y_max = data.unnormalize_target(1.0);
                    network.epsilon = 10000 / (y_max - y_min);
                    
                    %{
                    layers = [data.num_features 35 15 10 3];
                    network = ClassNet(layers);
                    classes = data.create_classes(3);
                    data_train = data.encode_targets(data_train, classes);
                    data_test = data.encode_targets(data_test, classes);
                    obj.display_model(network);
                    %}
                    
                    fprintf('\nTraining Model.\n');
                    num_iters = 200;
                    batch_size = 10;
                    %results = network.train(data_train, data_test, SGD(0.1, 0.9, 0.5), num_iters, batch_size, @obj.display_training);
                    %results = network.train(data_train, data_test, AdaGrad(0.1, 0.5), num_iters, batch_size, @obj.display_training);
                    results = network.train(data_train, data_test, AdaDelta(0.5), num_iters, batch_size, @obj.display_training);
                    obj.plot(data, network, results);
                    
                    fprintf('\nEvaluating model.\n');
                    [loss, acc] = network.evaluate(data_test);
                    obj.display_evaluation(loss, acc);    
            end
        end
        function display_data(obj, data, data_train, data_test)
            fprintf('Data Source: %s\n', data.name);
            fprintf('Total features: %d\n', data.num_features);
            fprintf('Total entries: %d\n', data.num_entries);
            fprintf('Training entries: %d\n', length(data_train));
            fprintf('Test entries: %d\n', length(data_test));
        end
        function display_model(obj, network)
            fprintf('Type: Feedforward Neural Network\n');
            fprintf('Objective: %s\n', network.name);
            fprintf('Layers:\n');
            n_layers = length(network.layers);
            for i = 1:n_layers
                fprintf('\t%d: %d units\n', i, network.layers(i))
            end
        end
        function display_training(obj, results)
            results = cell2mat(cellfun(@(x) cell2mat(x), results, 'un', 0));
            fprintf('[%4d] training [loss=%8.6f acc=%4.2f] validating [loss=%8.6f acc=%4.2f]\n', results(end,:));
            
            gcf;
            subplot(2, 1, 1);
            plot(results(:,1), results(:,3), 'r', results(:,1), results(:,5), 'g');
            ylabel('Accuracy');
            legend('Training Data', 'Test Data', 'Location', 'NorthEastOutside');
            grid('on');
            subplot(2, 1, 2);
            plot(results(:,1), results(:,2), 'r', results(:,1), results(:,4), 'g');
            legend('Training Data', 'Test Data', 'Location', 'NorthEastOutside');
            xlabel('Iteration');
            ylabel('Loss');
            grid('on');
            drawnow();
        end
        function display_evaluation(obj, loss, acc)
            fprintf('Results: [loss=%8.6f acc=%4.2f]\n', loss, acc*100.0);
        end
        function plot(obj, data, network, results)
            results = cell2mat(cellfun(@(x) cell2mat(x), results, 'un', 0));
            
            clf();
            figure(1);
            plot(results(:,1), results(:,3), 'r', results(:,1), results(:,5), 'g');
            title(data.name);
            xlabel('Iteration');
            ylabel('Accuracy');
            legend('Training Data', 'Test Data', 'Location', 'southeast');
            grid('on');
            file_path = sprintf('fig_%s_%s_acc.jpg', lower(data.name), lower(network.name));
            saveas(gcf, file_path);
            close(gcf)

            figure(2);
            plot(results(:,1), results(:,2), 'r', results(:,1), results(:,4), 'g');
            title(data.name);
            xlabel('Iteration');
            ylabel('Loss');
            legend('Training Data', 'Test Data', 'Location', 'northeast');
            grid('on');
            file_path = sprintf('fig_%s_%s_loss.jpg', lower(data.name), lower(network.name));
            saveas(gcf, file_path);
            close(gcf)
            
            figure(3);
            plot(results(:,1), results(:,2), 'r', results(:,1), results(:,4), 'g');
            title(data.name);
            xlabel('Iteration');
            ylabel('Loss');
            legend('Training Data', 'Test Data', 'Location', 'northeast');
            set(gca, 'YScale', 'log');
            grid('on');
            file_path = sprintf('fig_%s_%s_log_loss.jpg', lower(data.name), lower(network.name));
            saveas(gcf, file_path);
            close(gcf)
        end
    end
end