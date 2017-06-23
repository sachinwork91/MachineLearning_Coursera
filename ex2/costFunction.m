function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;


grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
first_part=log(sigmoid(X*theta)).*(-y);

second_part=(1-y).*log(1-sigmoid(X*theta)); 

J=(1/m)*sum(first_part-second_part);

 
 
 % Finding the gradient 
 % In this part
 % when we apply sigmoid function i.e hypothesis function we get result of 100*1 vector which is nothing but he result determined by out hypothesis function
 % Now we subtract that from the actual result to computer the cost or also reffered as Loss. 
 % So the result is 100*1 vector again
 % Now to computer the gradient here we are performing the derivatives which is basically that for each cost determinied we will try to identify the new theta for a training example
 % since we have 100 training examples 
 %	[1 2 3 4 5 6 ] [1 x1 x2] 
 %                 [1 x3 x4]
 %				   [1 x5 x6]
 %  so we compute theta(0) by mulitplying and adding all the cost with 1(bais unit), so we are basically taking all the training examples into consideration
 %  Now we compute the second column of our resultant matrix(theta(1)), (Note: we will be dividing the result by m as per the formula )
 %  so the computation is multiply and add all the column 2 data i.e 
 %  we are multiplying [1 2 3 4 5 6 ][x1
 %									  x3
 %                                     x5]
 %	After this we get contribution of first set of parameters from our training example and similarlly we calculate further		
 
grad1=(1/m)*((sigmoid(X*theta)-y)'*X);

grad=grad1;






% =============================================================

end
