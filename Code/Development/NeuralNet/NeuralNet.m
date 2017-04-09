classdef NeuralNet < handle
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
            n_layers = size(obj.layers, 2);
            obj.weights = cell(n_layers-1, 1);
            obj.biases = cell(n_layers-1, 1);
            for i = 2:n_layers
                obj.weights{i-1} = randn(obj.layers(i), obj.layers(i-1));
                obj.biases{i-1} = randn(obj.layers(i), 1);
            end
            
        end
        function reset(obj)
            % re-initialize weights and biases
            obj.weights = {};
            obj.biases = {};
            n_layers = size(obj.layers, 2);
            for i = 2:n_layers
                obj.weights{end+1} = randn(obj.layers(i), obj.layers(i-1));
                obj.biases{end+1} = randn(obj.layers(i), 1);
            end
        end
        function results = train(obj, data_train, data_test, num_iters, batch_size)
            results = {};
            
            best_loss = Inf;
            best_W = obj.weights;
            best_b = obj.biases;
            for i = 1:num_iters
                % gradient descent
                obj.gradient_descent(data_train, batch_size);
                
                [train_loss, train_acc] = obj.evaluate(data_train);
                [test_loss, test_acc] = obj.evaluate(data_test);
                if test_loss < best_loss
                    best_loss = test_loss;
                    best_W = obj.weights;
                    best_b = obj.biases;
                end
                results{end+1} = {i, train_loss, train_acc, test_loss, test_acc};
                fprintf('[%4d] training [loss=%8.6f acc=%4.2f] validating [loss=%8.6f acc=%4.2f]\n', i, train_loss, train_acc*100.0, test_loss, test_acc*100.0);
            end
            
            obj.weights = best_W;
            obj.biases = best_b;
        end
        function gradient_descent(obj, data, batch_size)
            eta = 0.1;
            [grad_W, grad_b] = obj.get_batch_gradient(data);
            n_layers = size(obj.layers, 2);
            for i = 1:n_layers-1
                w = obj.weights{i};
                b = obj.biases{i};
                gw = grad_W{i};
                gb = grad_b{i};
                obj.weights{i} = w - (eta * gw);
                obj.biases{i} = b - (eta * gb);
            end
        end
        function [grad_W, grad_b] = get_batch_gradient(obj, data)
            n_layers = size(obj.layers, 2);
            
            % initialize to zero
            batch_grad_W = cell(n_layers-1, 1);
            batch_grad_b = cell(n_layers-1, 1);
            for i = 1:n_layers-1
                batch_grad_W{i} = zeros(size(obj.weights{i}));
                batch_grad_b{i} = zeros(size(obj.biases{i}));
            end
            
            % sum the gradients for each point
            n_data = size(data, 1);
            for i = 1:n_data
                x = data{i,1};
                t = data{i,2};
                [grad_W, grad_b] = obj.back_propagation(x, t);
                for j = 1:n_layers-1
                    batch_grad_W{j} = batch_grad_W{j} + grad_W{j};
                    batch_grad_b{j} = batch_grad_b{j} + grad_b{j};
                end
            end

            % average the batch gradient
            for j = 1:n_layers-1
                batch_grad_W{j} = batch_grad_W{j} / n_data;
                batch_grad_b{j} = batch_grad_b{j} / n_data;
            end

            grad_W = batch_grad_W;
            grad_b = batch_grad_b;
        end
        function [grad_W, grad_b] = back_propagation(obj, x, t)
            n_layers = size(obj.layers, 2);
            
            % forward pass
            ws = obj.weights;
            bs = obj.biases;
            zs = cell(n_layers-1, 1);
            hs = cell(n_layers, 1);
            hs{1} = x;
            n_layers = size(obj.layers, 2);
            for i = 1:n_layers-1
                zs{i} = ws{i} * hs{i} + bs{i};
                hs{i+1} = obj.activation(zs{i});
            end
            y = hs{end};
            
            % backward pass
            grad_W = cell(n_layers-1, 1);
            grad_b = cell(n_layers-1, 1);
            delta_h = obj.error_deriv(y, t);
            for i = 1:n_layers-1
                delta_h = delta_h .* obj.activation_deriv(zs{end+1-i});
                grad_W{end+1-i} = delta_h * hs{end+1-i-1}.';
                grad_b{end+1-i} = delta_h;
                delta_h = ws{end+1-i}.' * delta_h;
            end
        end
        function y = predict(obj, x)
            n_layers = size(obj.layers, 2);
            
            h = x;
            for i = 1:n_layers-1
                w = obj.weights{i};
                b = obj.biases{i};
                z = w * h + b;
                h = obj.activation(z);
            end
            y = h;
        end
        function [loss, acc] = evaluate(obj, data)
            loss = 0.0;
            correct = 0.0;
            n_data = size(data, 1);
            for i = 1:n_data
                x = data{i,1};
                t = data{i,2};
                
                % make a prediction for the current data point
                y = obj.predict(x);
                
                % compute the error of the prediction
                loss = loss + obj.error(y, t);
                
                % check if prediction matches truth
                if obj.is_match(y, t)'
                    correct = correct + 1.0;
                end
            end
            
            loss = loss / n_data;
            acc = correct / n_data;
        end
        function value = activation(obj, z)
            value = 0.0;
        end
        function value = activation_deriv(obj, z)
            value = 0.0;
        end
        function value = error(obj, z)
            value = 0.0;
        end
        function value = error_deriv(obj, z)
            value = 0.0;
        end
        function value = is_match(obj, y, t)
            value = false;
        end
    end
end