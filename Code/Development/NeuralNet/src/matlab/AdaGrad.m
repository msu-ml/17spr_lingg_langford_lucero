classdef AdaGrad < GradientDescent
    methods
        function obj = AdaGrad(learning_rate)
            obj = obj@GradientDescent(learning_rate);
        end
        function optimize(obj, network, dataset, batch_size)
            eta = obj.learning_rate;
            eps = 1e-8;
        
            % Randomly shuffle the training data and split it into batches.
            dataset.shuffle();
            
            n_layers = length(network.layers);
            mem_gW = cell(n_layers-1, 1);
            mem_gb = cell(n_layers-1, 1);
            for i = 1:n_layers-1
                mem_gW{i} = zeros(size(network.weights{i}));
                mem_gb{i} = zeros(size(network.biases{i}));
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
                    mgw = mem_gW{j};
                    mgb = mem_gb{j};
                    
                    mgw = mgw + gw.^2;
                    mgb = mgb + gb.^2;
                    dw = -((gw * eta) ./ (mgw + eps).^0.5);
                    db = -((gb * eta) ./ (mgb + eps).^0.5);
                    network.weights{j} = w + dw;
                    network.biases{j} = b + db;
                    
                    mem_gW{j} = mgw;
                    mem_gb{j} = mgb;
                end
            end
        end
    end
end