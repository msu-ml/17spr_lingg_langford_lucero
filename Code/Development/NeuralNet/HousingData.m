classdef HousingData
    properties
        name
        fields
        data
        data_min
        data_max
        num_entries
        num_features
    end
    methods
        function obj = HousingData(file_path, name)
            obj.name = name;
            obj.fields = {};
            obj.data = {};
            obj.data_min = {};
            obj.data_max = {};
            
            % read data from preprocessed csv file.
            tbl = readtable(file_path);
            fields = tbl.Properties.VariableNames;
            data = table2array(tbl(:,:));
            
            % read unnormalized data bounds from preprocessed csv file.
            bounds_file_path = strrep(file_path, '.csv', '_bounds.csv');
            bounds_tbl = readtable(bounds_file_path);
            bounds_X = table2array(bounds_tbl(:,1:end-1));
            bounds_y = table2array(bounds_tbl(:,end));
            obj.data_min = {bounds_X(1,:), bounds_y(1)};
            obj.data_max = {bounds_X(2,:), bounds_y(2)};
            
            % separate the target field from the rest.
            [X, y, X_fields, y_fields] = obj.separate_targets(data, fields, fields{end});
           
            % reshape data and store it
            obj.fields{end+1} = {X_fields, y_fields};
            for i = 1:size(X, 1)
                obj.data = [obj.data; {X(i,:)', y(i,:)'}];
            end
            obj.num_entries = size(obj.data, 1);
            obj.num_features = size(obj.data{1}, 1);
        end
        function [X, y, X_fields, y_fields] = separate_targets(obj, data, fields, target_field)
            target_column = find(~cellfun('isempty', strfind(fields, target_field)));
            idx = true(1, size(data, 2));
            idx(target_column) = false;
            X = data(:,idx);
            y = data(:,~idx);
            X_fields = fields(idx);
            y_fields = fields(~idx);
        end
        function [data_train, data_test] = split_data(obj, a, b)
            num_train = floor(obj.num_entries * a / (a + b));
            data_train = obj.data(1:num_train+1,:);
            data_test = obj.data(num_train+2:end,:);
        end
        function classes = create_classes(obj, num_classes)
            classes = [];
            targets = sort(cell2mat(obj.data(:,2)));
            num_targets = size(targets, 1);
            batch_size = floor(num_targets / num_classes);
            num_targets = num_classes * batch_size;
            for i = 1:batch_size:num_targets
                batch = targets(i:i+batch_size);
                classes = [classes, batch(1)];
            end
        end
        function t = encode_target(obj, target, classes)
            target_class = 0;
            n_classes = size(classes, 2);
            for i = 1:n_classes
                if target >= classes(i)
                    target_class = i;
                end
            end
            t = zeros(n_classes, 1);
            t(target_class) = 1.0;
        end
        function encoded_data = encode_targets(obj, data, classes)
            encoded_data = data;
            y = data(:,2);
            n_data = size(data, 1);
            for i = 1:n_data
                encoded_data{i,2} = obj.encode_target(y{i}, classes);
            end 
        end
        function value = unnormalize_target(obj, value)
            y_max = obj.data_max{2};
            y_min = obj.data_min{2};
            value = ((y_max - y_min) * value) + y_min;
        end
    end
end