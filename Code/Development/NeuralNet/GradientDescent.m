classdef GradientDescent < handle
    properties
        learning_rate
    end
    methods
        function obj = GradientDescent(learning_rate)
            obj.learning_rate = learning_rate;
        end
        function optimize(obj, network, dataset, batch_size)
            eta = obj.learning_rate;
            [grad_W, grad_b] = obj.get_batch_gradient(network, dataset);
            n_layers = length(network.layers);
            for i = 1:n_layers-1
                w = network.weights{i};
                b = network.biases{i};
                gw = grad_W{i};
                gb = grad_b{i};
                network.weights{i} = w - (eta * gw);
                network.biases{i} = b - (eta * gb);
            end
        end
        function [grad_W, grad_b] = get_batch_gradient(obj, network, dataset)
            n_layers = length(network.layers);
            
            % initialize to zero
            batch_grad_W = cell(n_layers-1, 1);
            batch_grad_b = cell(n_layers-1, 1);
            for i = 1:n_layers-1
                batch_grad_W{i} = zeros(size(network.weights{i}));
                batch_grad_b{i} = zeros(size(network.biases{i}));
            end
            
            % sum the gradients for each point
            data = dataset.get_data();
            targets = dataset.get_targets();
            n_entries = dataset.num_entries;
            for i = 1:n_entries
                x = data(i,:);
                t = targets(i,:);
                [grad_W, grad_b] = network.back_propagation(x, t);
                for j = 1:n_layers-1
                    batch_grad_W{j} = batch_grad_W{j} + grad_W{j};
                    batch_grad_b{j} = batch_grad_b{j} + grad_b{j};
                end
            end

            % average the batch gradient
            for j = 1:n_layers-1
                batch_grad_W{j} = batch_grad_W{j} / n_entries;
                batch_grad_b{j} = batch_grad_b{j} / n_entries;
            end

            grad_W = batch_grad_W;
            grad_b = batch_grad_b;
        end
    end
end