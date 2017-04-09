classdef GradientDescent < handle
    properties
        learning_rate
    end
    methods
        function obj = GradientDescent(learning_rate)
            obj.learning_rate = learning_rate;
        end
        function optimize(obj, network, data, batch_size)
            eta = obj.learning_rate;
            [grad_W, grad_b] = obj.get_batch_gradient(network, data);
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
        function [grad_W, grad_b] = get_batch_gradient(obj, network, data)
            n_layers = length(network.layers);
            
            % initialize to zero
            batch_grad_W = cell(n_layers-1, 1);
            batch_grad_b = cell(n_layers-1, 1);
            for i = 1:n_layers-1
                batch_grad_W{i} = zeros(size(network.weights{i}));
                batch_grad_b{i} = zeros(size(network.biases{i}));
            end
            
            % sum the gradients for each point
            n_data = length(data);
            for i = 1:n_data
                x = data{i,1};
                t = data{i,2};
                [grad_W, grad_b] = network.back_propagation(x, t);
                for j = 1:n_layers-1
                    batch_grad_W{j} = batch_grad_W{j} + grad_W{j};
                    batch_grad_b{j} = batch_grad_b{j} + grad_b{j};
                end
            end

            % average the batch gradient
            for j = 1:n_layers-1
                batch_grad_W{j} = batch_grad_W{j} / n_data;
                batch_grad_b{j} = batch_grad_b{j} / n_data;
            end

            grad_W = batch_grad_W;
            grad_b = batch_grad_b;
        end
    end
end