classdef SGD < GradientDescent
    properties
        momentum
        l1_regularization
        l2_regularization
    end
    methods
        function obj = SGD(learning_rate, momentum, l1_regularization, l2_regularization)
            obj = obj@GradientDescent(learning_rate);
            obj.momentum = momentum;
            obj.l1_regularization = l1_regularization;
            obj.l2_regularization = l2_regularization;
        end
        function optimize(obj, network, dataset, batch_size)
            eta = obj.learning_rate;
            rho = obj.momentum;
            lambda1 = obj.l1_regularization;
            lambda2 = obj.l2_regularization;
        
            % Randomly shuffle the training data and split it into batches.
            dataset.shuffle();
            
            n_layers = length(network.layers);
            mem_dW = cell(n_layers-1, 1);
            mem_db = cell(n_layers-1, 1);
            for i = 1:n_layers-1
                mem_dW{i} = zeros(size(network.weights{i}));
                mem_db{i} = zeros(size(network.biases{i}));
            end
            
            batches = dataset.make_batches(batch_size);
            n_batches = length(batches);
            for i = 1:n_batches
                [grad_W, grad_b] = obj.get_batch_gradient(network, batches{i});
                n_layers = length(network.layers);
                for j = 1:n_layers-1
                    w = network.weights{j};
                    b = network.biases{j};
                    gw = grad_W{j};
                    gb = grad_b{j};
                    mdw = mem_dW{j};
                    mdb = mem_db{j};
                    
                    l1_term = eta * lambda1 / batch_size;
                    l2_term = eta * lambda2 / batch_size;
                    dw = (rho * mdw) + (eta * gw);
                    db = (rho * mdb) + (eta * gb);
                    network.weights{j} = ((1- l2_term) * w) - (l2_term * sign(w)) - dw;
                    network.biases{j} = b - db;
                    
                    mem_dW{j} = dw;
                    mem_db{j} = db;
                end
            end
        end
    end
end