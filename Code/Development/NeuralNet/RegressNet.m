classdef RegressNet < NeuralNet
    properties
        epsilon
    end
    methods
        function obj = RegressNet(layers)
            obj = obj@NeuralNet(layers);
            obj.name = 'Regression'
            obj.epsilon = 1e-8;
        end
        function value = activation(obj, z)
            sigmoid = @(z) 1.0 ./ (1.0 + exp(-z));
            value = sigmoid(z);
        end
        function value = activation_deriv(obj, z)
            sigmoid = @(z) 1.0 ./ (1.0 + exp(-z));
            value = sigmoid(z) .* (1.0 - sigmoid(z));
        end
        function value = error(obj, y, t)
            value = (y - t) * (y - t);
        end
        function value = error_deriv(obj, y, t)
            value = y - t;
        end
        function value = is_match(obj, y, t)
            value = abs(y - t) <= obj.epsilon;
        end
    end
end