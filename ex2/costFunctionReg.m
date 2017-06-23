function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
size(theta)
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
first_part=log(sigmoid(X*theta)).*(-y);

second_part=(1-y).*log(1-sigmoid(X*theta)); 



J=(1/m)*sum(first_part-second_part) + (lambda/(2*m))*sum((theta(2:end).^2));


%Computing the first part of gradient descent 
% here we are computing the loss from sigmoid(X*theta)-y and then trasnform that to multiply with X matrix
% Again GD is basically the effect of each parameter of a matix 
% so  loss in our case is a matix of 118*1 we transpose it to get [1 2 3 4 5 .....128] (this is just to display the matix content after transpose) and then multiply by X
% [1 2 3 4 5 .....128][1 x1 x2 .......x28]  ---1st row of training example
					  [1 x2 x3 .......x28]  ---2nd row of training example 
					  [1 x2 x3 .......x28]  ---3rd row of training example
					  [1 x2 x3 .......x28]  ---4th row of training example
					  
grad=(1/m)*((sigmoid(X*theta) - y )' * X);
size(grad)
grad=grad';
% by the above computation we get the result for in 1*28 form 
% however we have to return 28*1
% So after transforming our grad will look like 
% [1
%  2
%  3
%  4
%  .
%  .
%  .28]

%After this we need to add the reqularization term for the theats other the zero. (since in octave indexes start from 1 , we start adding regularization from index 2 )
grad(2:end)=grad(2:end).+((lambda/m)*(theta(2:end)));




%grad(1)=(1/m)*(sigmoid(X*theta)-y).*X(:,1);
%for i = 2:size(theta, 1)
%    grad(i) = 1 / m * sum((sigmoid(X*theta) - y) .* X(:, i)) + lambda / m * theta(i);
%end


% =============================================================

end
