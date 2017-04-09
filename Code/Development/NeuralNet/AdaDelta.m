classdef AdaDelta < GradientDescent
    properties
        scale
    end
    methods
        function obj = AdaDelta(scale)
            obj = obj@GradientDescent(0.0);
            obj.scale = scale;
        end
        function optimize(obj, network, data, batch_size)
            rho = obj.scale;
            eps = 1e-8;
        
            % Randomly shuffle the training data and split it into batches.
            n_data = size(data, 1);
            data = data(randperm(n_data),:);
            
            n_layers = length(network.layers);
            mem_dW = cell(n_layers-1, 1);
            mem_db = cell(n_layers-1, 1);
            mem_gW = cell(n_layers-1, 1);
            mem_gb = cell(n_layers-1, 1);
            for i = 1:n_layers-1
                mem_dW{i} = zeros(size(network.weights{i}));
                mem_db{i} = zeros(size(network.biases{i}));
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
                    mdw = mem_dW{j};
                    mdb = mem_db{j};
                    mgw = mem_gW{j};
                    mgb = mem_gb{j};
                    
                    mgw = ((rho * mgw) + ((1 - rho) * gw.^2));
                    mgb = ((rho * mgb) + ((1 - rho) * gb.^2));
                    dw = -((gw .* (mdw + eps).^0.5) ./ (mgw + eps).^0.5);
                    db = -((gb .* (mdb + eps).^0.5) ./ (mgb + eps).^0.5);
                    mdw = ((rho * mdw) + ((1 - rho) * dw.^2));
                    mdb = ((rho * mdb) + ((1 - rho) * db.^2));
                    network.weights{j} = w + dw;
                    network.biases{j} = b + db;
                    
                    mem_dW{j} = mdw;
                    mem_db{j} = mdb;
                    mem_gW{j} = mgw;
                    mem_gb{j} = mgb;
                end
            end
        end
    end
end