classdef Experiment < handle
    properties
        sources
    end
    methods
        function obj = Experiment()
            fprintf('\nLoading data.\n');
            obj.sources = {};
            %obj.sources{end+1} = HousingData('Data/nashville_processed.csv', 'Nashville');
            %obj.sources{end+1} = HousingData('Data/kingcounty_processed.csv', 'KingCounty');
            %obj.sources{end+1} = HousingData('Data/redfin_processed.csv', 'GrandRapids');
            obj.sources{end+1} = HousingData('Data/art_processed.csv', 'ART');
        end
        function run(obj)
            n_sources = length(obj.sources);
            for i = 1:n_sources
                source = obj.sources{i};
                
                fprintf('\nData ----------------------------------\n');
                [dataset_train, dataset_test] = source.to_dataset().split(2.0/3.0);
                obj.display_data(source, dataset_train, dataset_test);
                
                fprintf('\nModel ---------------------------------\n');
                layers = [dataset_train.num_features 35 15 10 1];
                network = RegressNet(layers);
                y_min = source.unnormalize_target(0.0);
                y_max = source.unnormalize_target(1.0);
                network.epsilon = 10000 / (y_max - y_min);
                obj.display_model(network);

                %{
                layers = [dataset_train.num_features 35 15 10 3];
                network = ClassNet(layers);
                classes = dataset_train.create_classes(3);
                dataset_train.encode_targets(classes);
                dataset_test.encode_targets(classes);
                obj.display_model(network);
                %}

                fprintf('\nTraining Model.\n');
                num_iters = 500;
                batch_size = 10;
                results = network.train(dataset_train, dataset_test, SGD(0.1, 0.9, 0.5), num_iters, batch_size, @obj.display_training);
                %results = network.train(dataset_train, dataset_test, AdaGrad(0.1, 0.5), num_iters, batch_size, @obj.display_training);
                %results = network.train(dataset_train, dataset_test, AdaDelta(0.5), num_iters, batch_size, @obj.display_training);
                obj.plot(source, network, results);

                fprintf('\nEvaluating model.\n');
                [loss, acc] = network.evaluate(dataset_test);
                obj.display_evaluation(loss, acc);
            end
        end
        function display_data(obj, source, data_train, data_test)
            fprintf('Data Source: %s\n', source.name);
            fprintf('Total features: %d\n', data_train.num_features);
            fprintf('Total entries: %d\n', data_train.num_entries + data_test.num_entries);
            fprintf('Training entries: %d\n', data_train.num_entries);
            fprintf('Test entries: %d\n', data_test.num_entries);
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
        function plot(obj, source, network, results)
            results = cell2mat(cellfun(@(x) cell2mat(x), results, 'un', 0));
            
            clf();
            figure(1);
            plot(results(:,1), results(:,3), 'r', results(:,1), results(:,5), 'g');
            title(source.name);
            xlabel('Iteration');
            ylabel('Accuracy');
            legend('Training Data', 'Test Data', 'Location', 'southeast');
            grid('on');
            file_path = sprintf('fig_%s_%s_acc.jpg', lower(source.name), lower(network.name));
            saveas(gcf, file_path);
            close(gcf)

            figure(2);
            plot(results(:,1), results(:,2), 'r', results(:,1), results(:,4), 'g');
            title(source.name);
            xlabel('Iteration');
            ylabel('Loss');
            legend('Training Data', 'Test Data', 'Location', 'northeast');
            grid('on');
            file_path = sprintf('fig_%s_%s_loss.jpg', lower(source.name), lower(network.name));
            saveas(gcf, file_path);
            close(gcf)
            
            figure(3);
            plot(results(:,1), results(:,2), 'r', results(:,1), results(:,4), 'g');
            title(source.name);
            xlabel('Iteration');
            ylabel('Loss');
            legend('Training Data', 'Test Data', 'Location', 'northeast');
            set(gca, 'YScale', 'log');
            grid('on');
            file_path = sprintf('fig_%s_%s_loss_log.jpg', lower(source.name), lower(network.name));
            saveas(gcf, file_path);
            close(gcf)
        end
    end
end