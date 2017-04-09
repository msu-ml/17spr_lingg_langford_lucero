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
            n_data = size(data, 1);
            eta = obj.learning_rate;
            rho = obj.momentum;
            lambda = obj.regularization;
        
            % term for regularizing the weights.
            reg_decay = (1.0 - (eta * lambda / n_data));
        
            % Randomly shuffle the training data and split it into batches.
            data = data(randperm(n_data),:);
            
            n_data = n_data - mod(n_data, batch_size);
            for i = 1:batch_size:n_data
                batch = data(i:i+batch_size-1,:);
                [grad_W, grad_b] = obj.get_batch_gradient(network, batch);
                n_layers = size(network.layers, 2);
                for j = 1:n_layers-1
                    w = network.weights{j};
                    b = network.biases{j};
                    gw = grad_W{j};
                    gb = grad_b{j};
                    network.weights{j} = w - (eta * gw);
                    network.biases{j} = b - (eta * gb);
                end
            end


            %{
                    mem_dW = [np.zeros(w.shape) for w in network.weights]
                    mem_db = [np.zeros(b.shape) for b in network.biases]
                    for i in xrange(0, len(data), batch_size):
                        batch = data[i:i+batch_size]
                        grad_W, grad_b = self.get_batch_gradient(network, batch)
                        delta_W = [((rho * mdw) + (eta * gw)) for mdw, gw in zip(mem_dW, grad_W)]
                        delta_b = [((rho * mdb) + (eta * gb)) for mdb, gb in zip(mem_db, grad_b)]
                        mem_dW = [dw for dw in delta_W]
                        mem_db = [db for db in delta_b]
                        network.weights = [(reg_decay * w - dw) for w, dw in zip(network.weights, delta_W)]
                        network.biases = [(b - db) for b, db in zip(network.biases, delta_b)]
            %}
        end
    end
end