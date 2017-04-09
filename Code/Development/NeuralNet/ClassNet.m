classdef ClassNet < NeuralNet
    methods
        function obj = ClassNet(layers)
            obj = obj@NeuralNet(layers);
            obj.name = 'Classfication'
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
            value = -sum((t .* log(y)) + (1 - t) .* log(1.0 - y));
        end
        function value = error_deriv(obj, y, t)
            value = y - t;
        end
        function value = is_match(obj, y, t)
            [~, y_argmax] = max(y);
            [~, t_argmax] = max(t);
            value = (y_argmax == t_argmax);
        end
    end
end