classdef AdaGrad < GradientDescent
    properties
        regularization
    end
    methods
        function obj = AdaGrad(learning_rate, regularization)
            obj = obj@GradientDescent(learning_rate);
            obj.regularization = regularization;
        end
        function optimize(obj, network, data, batch_size)
            eta = obj.learning_rate;
            lambda = obj.regularization;
            eps = 1e-8;
        
            % term for regularizing the weights.
            n_data = size(data, 1);
            reg_decay = (1.0 - (eta * lambda / n_data));
        
            % Randomly shuffle the training data and split it into batches.
            data = data(randperm(n_data),:);
            
            n_layers = length(network.layers);
            mem_gW = cell(n_layers-1, 1);
            mem_gb = cell(n_layers-1, 1);
            for i = 1:n_layers-1
                mem_gW{i} = zeros(size(network.weights{i}));
                mem_gb{i} = zeros(size(network.biases{i}));
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
                    mgw = mem_gW{j};
                    mgb = mem_gb{j};
                    
                    mgw = mgw + gw.^2;
                    mgb = mgb + gb.^2;
                    dw = -((gw * eta) ./ (mgw + eps).^0.5);
                    db = -((gb * eta) ./ (mgb + eps).^0.5);
                    network.weights{j} = reg_decay * w + dw;
                    network.biases{j} = b + db;
                    
                    mem_gW{j} = mgw;
                    mem_gb{j} = mgb;
                end
            end
        end
    end
end