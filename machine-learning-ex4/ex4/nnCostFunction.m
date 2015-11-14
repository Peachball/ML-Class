function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%nn_params refers to the capital theta
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X = [ones(size(X,1),1), X];
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
z2 = X * Theta1';
a2 = sigmoid(z2); %each row is from a different input set (example)
a2 = [ones(size(a2,1),1), a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
cursum = 0;
for i = 1:num_labels
    cursum = cursum + sum(-logical(y==i).*log(a3(:,i)) - (1-logical(y==i)).*log(1-a3(:,i)));
end
regular = sum(nn_params.^2) - sum(Theta1(:,1).^2) - sum(Theta2(:,1).^2);


J = cursum / m + regular * lambda / 2 / m;
%Cost function works correctly

%Gradient finder code here
D2 = zeros(size(Theta2));
D1 = zeros(size(Theta1));
for ex = 1:size(X,1)
    %Foward Propogation
    a1 = X(ex,:)';
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1;a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
    answer = zeros(num_labels,1);
    answer(y(ex)) = 1;
    delta3 = a3 - answer;
    
    delta2 = Theta2' * delta3;
    delta2 = delta2(2:end); %Remove delta2(0)
    delta2 = delta2.*sigmoidGradient(z2);
    %Now delta2 and delta3 should be calculated and ready to go
    
    D2 = D2 + (delta3 * a2');
    D1 = D1 + (delta2 * a1');
end

Theta1_grad = D1 / m;
Theta2_grad = D2 / m;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
