classdef NeuralNet
    properties
        layers
        name
        weights
        biases
    end
    methods
        function obj = NeuralNet(layers)
            obj.layers = layers;
            obj.name = '';
            
            % initialize weights and biases
            n = size(obj.layers, 2);
            obj.weights = {};
            obj.biases = {};
            for i = 2:n
                obj.weights{end+1} = randn(obj.layers(i), obj.layers(i-1));
                obj.biases{end+1} = randn(obj.layers(i), 1);
            end
        end
        function reset(obj)
            % re-initialize weights and biases
            n = size(obj.layers, 2);
            obj.weights = {};
            obj.biases = {};
            for i = 2:n
                obj.weights{end+1} = randn(obj.layers(i), obj.layers(i-1));
                obj.biases{end+1} = randn(obj.layers(i), 1);
            end
        end
        function results = train(obj, data_train, data_test, num_iters, batch_size)
            results = {};
            for i = 1:num_iters
                % gradient descent
                obj.gradient_descent(data_train, batch_size);
                
                % evaluate training data
                % evaluate test data
            end
        end
        function gradient_descent(obj, data, batch_size)
            eta = 0.1;
            [grad_W, grad_b] = obj.get_batch_gradient(data_train);
            n = size(obj.weights, 2);
            for i = 1:n
                gw = grad_W{i}
                gb = grad_b{i}
                dw = -(eta * gw)
                db = -(eta * gb)
                obj.weights{i} = w + dw
                obj.weights{i} = b + db
            end
        end
        function get_batch_gradient(obj, data_train)
        end
    end
end