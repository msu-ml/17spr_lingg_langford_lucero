classdef SGD < GradientDescent
    properties
        momentum
        regularization
    end
    methods
        function obj = SGD(learning_rate, momentum, regularization)
            obj = obj@GradientDescent(learning_rate);
            obj.momentum = momentum;
            obj.regularization = regularization;
        end
        function optimize(obj, network, data, batch_size)
            eta = obj.learning_rate;
            rho = obj.momentum;
            lambda = obj.regularization;
        
            % term for regularizing the weights.
            n_data = length(data);
            reg_decay = (1.0 - (eta * lambda / n_data));
        
            % Randomly shuffle the training data and split it into batches.
            data = data(randperm(n_data),:);
            
            n_layers = length(network.layers);
            mem_dW = cell(n_layers-1, 1);
            mem_db = cell(n_layers-1, 1);
            for i = 1:n_layers-1
                mem_dW{i} = zeros(size(network.weights{i}));
                mem_db{i} = zeros(size(network.biases{i}));
            end
            
            n_data = n_data - mod(n_data, batch_size);
            for i = 1:batch_size:n_data
                batch = data(i:i+batch_size-1,:);
                [grad_W, grad_b] = obj.get_batch_gradient(network, batch);
                n_layers = length(network.layers);
                for j = 1:n_layers-1
                    w = network.weights{j};
                    b = network.biases{j};
                    gw = grad_W{j};
                    gb = grad_b{j};
                    mdw = mem_dW{j};
                    mdb = mem_db{j};
                    
                    dw = (rho * mdw) + (eta * gw);
                    db = (rho * mdb) + (eta * gb);
                    network.weights{j} = reg_decay * w - dw;
                    network.biases{j} = b - db;
                    
                    mem_dW{j} = dw;
                    mem_db{j} = db;
                end
            end
        end
    end
end