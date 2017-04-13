classdef HousingData < handle
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
            obj.fields = {X_fields, y_fields};
            obj.data = {X, y};
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
        function value = unnormalize_target(obj, value)
            y_max = obj.data_max{2};
            y_min = obj.data_min{2};
            value = ((y_max - y_min) * value) + y_min;
        end
        function dataset = to_dataset(obj)
            dataset = Dataset(obj.data{1}, obj.data{2});
        end
    end
end