classdef Dataset < handle
    properties
        num_entries
        num_features
        dataset
    end
    methods
        function obj = Dataset(data, targets)
            obj.num_entries = size(data, 1);
            obj.num_features = size(data, 2);
            obj.dataset = [data, targets];
        end
        function data = get_data(obj)
            f = obj.num_features;
            data = obj.dataset(:,1:f);
        end
        function targets = get_targets(obj)
            f = obj.num_features;
            targets = obj.dataset(:,f+1:end);
        end
        function shuffle(obj)
            obj.dataset = obj.dataset(randperm(obj.num_entries),:);
        end
        function [dataset1, dataset2] = split(obj, ratio)
            n = floor(obj.num_entries * ratio);
            f = obj.num_features;
            dataset1 = Dataset(obj.dataset(1:n,1:f), obj.dataset(1:n,f+1:end));
            dataset2 = Dataset(obj.dataset(n+1:end,1:f), obj.dataset(n+1:end,f+1:end));
        end
        function classes = create_classes(obj, num_classes)
            classes = zeros(1, num_classes);
            targets = sort(obj.get_targets());
            batch_size = floor(obj.num_entries / num_classes);
            n_targets = batch_size * num_classes;
            j = 1;
            for i = 1:batch_size:n_targets
                batch = targets(i:i+batch_size);
                classes(j) = batch(1);
                j = j + 1;
            end
        end
        function t = encode_target(obj, target, classes)
            target_class = 0;
            n_classes = length(classes);
            for i = 1:n_classes
                if target >= classes(i)
                    target_class = i;
                end
            end
            t = zeros(1, n_classes);
            t(target_class) = 1.0;
        end
        function encode_targets(obj, classes)
            targets = obj.get_targets();
            enc_targets = zeros(obj.num_entries, length(classes));
            n_entries = obj.num_entries;
            for i = 1:n_entries
                enc_targets(i,:) = obj.encode_target(targets(i), classes);
            end
            obj.dataset = [obj.get_data(), enc_targets];
        end
        function batches = make_batches(obj, batch_size)
            n_entries = obj.num_entries - mod(obj.num_entries, batch_size);
            f = obj.num_features;
            batches = cell(1, n_entries / batch_size);
            j = 1;
            for i = 1:batch_size:n_entries
                batch_data = obj.dataset(i:i+batch_size,1:f);
                batch_targets = obj.dataset(i:i+batch_size,f+1:end);
                batches{j} = Dataset(batch_data, batch_targets);
                j = j + 1;
            end
        end
    end
end