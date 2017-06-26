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

%%%%%%Following are the size of the Matrix for Reference
%% size(X)=>5000x401
%% Size(Y)=>5000x1
%% Size(Theta1)=>25X401				hint: Theta1 is used in for Mapping Layer 1 to Layer 2  (no. of units in next layer) X (no.of units in current layer + 1) bias is added for 400 +1
%% Size(Theta2)=>10X26				Mapping from Layer 2 to Layer 3.....2nd layer has 26 units including the 1 extra for bias unit
%% Size(Cost_matrix)=>5000x10		This contains the cost, we have 5000 training examples and 10 classes so this matrix contains cost in identify each class for the for each training example
%% Size(delta_3k)=>5000X10			This is the error matrix for the Layer 3 , error for calculating 3 classes 
%% Size(delta_2k)=>5000X26			This is the error matrix for Layer 2,  identifyng the error in Layer 2 in calulating the value of units	
%% Size(triangular_delta1)=>25X401	This matrix is same as the Theta1_grad	we have 401(theta's) for 25 units in layer 2
%% Size(triangular_delta2)=>10X26	This matrix is same as the Theta1_grad  we have 26(units of layer 2 which will be multiplied with thetas) for 10 units in Layer 3
%%%%%%
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


% Add ones to the X data matrix
X = [ones(m, 1) X];
% Convert y from (1-10) class into num_labels vector
%Converting the Y vector into the corresponding vectors of the 10 classes
%y=[1 				Corresponds to y=[1 0 0 0 0 0 0 0 0 0 0 
%	2								  0 1 0 0 0 0 0 0 0 0 0  	
%	3								  0 0 1 0 0 0 0 0 0 0 0	
%	5								  0 0 0 0 1 0 0 0 0 0 0 	
%	5								  0 0 0 0 1 0 0 0 0 0 0 		
%	..]								  .  . . . . . .  . .  .  and so on 		
yd = eye(num_labels);% creates identity matix of num_labelsXnum_labels 
y = yd(y,:);		 % Maps y into the vector	
 
%Consider the first layer inputs as A1
A1=X;						
%size of X= 5000x401
%size of theta=5000x26, this 26 corresponds to the hidden layer units,1 extra because we will be adding bias unit
%Coverts to matrix of 5000 examples x 25 thetas
Z2=X*Theta1';
% Applying sigmoid function so that the results as between 0 and 1
A2=sigmoid(Z2);

%Adding one to the Computed units of Layer 2
% Add ones to the h1 data matrix
A2=[ones(m, 1) A2];
%size of A2 =5000X26
%Size of Theta2=10X26
%Now calculating the units of Layer 3 by multiplying the Theta2 and A2
% Converts to matrix of 
Z3=A2*Theta2';
% Sigmoid function converts to p between 0 to 1
A3=sigmoid(Z3);



% Compute cost
cost_matrix=(-y).*log(A3)-(1-y).*log(1-A3); % As y is now a matrix, so use dot product, unlike above
%Lets assume y is a vector containing results for any of 5 classes 
% y[1	not after performing the operation y=yd(y,:) it becomes [1 0 0 0 0 0					
%	2															 0 1 0 0 0 0 	
%   2															 0 1 0 0 0 0	
%   3]															 0 0 1 0 0 0]	
%now since we have y as a matrix we need dot product with the A3 i.e the final layer results 
%  		[1 0 0 0 0 0					
%		 0 1 0 0 0 0 	dotproduct qith A3
%   	 0 1 0 0 0 0	
%   	 0 0 1 0 0 0]	
% After performing this operation we get a Matrix of A3 i.e 5000X10


%J=(1/m)*sum(sum(cost_matrix));						Basically we are adding all the elements of the cost matrix
%
%
%
%
%
J=(1/m)*sum(cost_matrix(:))

%Implementing regularization
reg_term_for_theta1=sum((sum(Theta1(:,2:end).^2)));
reg_term_for_theta2=sum(sum(Theta2(:, 2:end) .^ 2));
Total_reg=(lambda / (2 * m))*(reg_term_for_theta1 + reg_term_for_theta2);

J=J + Total_reg;


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


    %Theta1_grad=0;
	%Theta2_grad=0;
%end
triangular_delta1=0;
triangular_delta2=0;

delta_3k=A3-y;  %Size of delta_3k=>5000X10
%'size of Theta2=> 10X26';
Z2=[ones(m,1),Z2];
delta_2k=delta_3k*Theta2.*sigmoidGradient(Z2);
delta_2k=delta_2k(:,2:end);       %we did not do this step for delta_3k as there was no bias unit, A3 and y corresponds to the results calculated
%size of delta_2=>10*
triangular_delta1=triangular_delta1+delta_2k'*A1;%(delta_2k'=>26*5000 and A1=>5000*401) A1 is the input 5000 training examples each with 400 pixels corresponding to images	
 
 
triangular_delta2=triangular_delta2+delta_3k'*A2; % delta_3k'=>10X5000 and A2=>5000*26


Theta1_grad=(1/m)*triangular_delta1;
Theta2_grad=(1/m)*triangular_delta2;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Performing Regularization on Gradients
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*(Theta1(:,2:end));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*(Theta2(:,2:end));


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end