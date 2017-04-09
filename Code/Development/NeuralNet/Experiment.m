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
            end
        end
        function display_data(obj, data, data_train, data_test)
            fprintf('Data Source: %s\n', data.name);
            fprintf('Total features: %d\n', data.num_features);
            fprintf('Total entries: %d\n', data.num_entries);
            fprintf('Training entries: %d\n', size(data_train, 1));
            fprintf('Test entries: %d\n', size(data_test, 1));
        end
    end
end