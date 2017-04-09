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
    end
end