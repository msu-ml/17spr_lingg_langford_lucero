classdef Experiment
    properties
        sources
    end
    methods
        function obj = Experiment()
            fprintf('\nLoading data.\n');
            obj.sources = { ...
                %HousingData('Data/nashville_processed.csv', 'Nashville'), ...
                %HousingData('Data/kingcounty_processed.csv', 'KingCounty'), ...
                %HousingData('Data/redfin_processed.csv', 'GrandRapids'), ...
                HousingData('Data/art_processed.csv', 'ART') ...
                          };
        end
        function run(obj)
            n_sources = size(obj.sources, 1);
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
                    obj.display_model(network);
                    
                    fprintf('\nTraining Model.');
                    num_iters = 50;
                    batch_size = 10;
                    results = network.train(data_train, data_test, num_iters, batch_size);
                    %obj.plot(data, network, results);
                    
                    fprintf('\nEvaluating model.');
                    %results = network.evaluate(data_test);
                    %obj.display_evaluation(results);
            end
        end
        function display_data(obj, data, data_train, data_test)
            fprintf('Data Source: %s\n', data.name);
            fprintf('Total features: %d\n', data.num_features);
            fprintf('Total entries: %d\n', data.num_entries);
            fprintf('Training entries: %d\n', size(data_train, 1));
            fprintf('Test entries: %d\n', size(data_test, 1));
        end
        function display_model(obj, network)
            fprintf('Type: Feedforward Neural Network\n');
            fprintf('Objective: %s\n', network.name);
            fprintf('Layers:\n');
            n_layers = size(network.layers, 2);
            for i = 1:n_layers
                fprintf('\t%d: %d units\n', i, network.layers(i))
            end
        end
    end
end