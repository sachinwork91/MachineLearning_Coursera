function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
fprintf('Preedicting the size of P\n');
size(p)
% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

%X is our feature matrix of size 5000*400 that means we have 5000 images and since each image is of 20X20 size we have 400 pixels 
%These 400 pixels acts a features. Now we have all_theta matix which is of 10*401 in size. 10 because we have 10 classes 
%and 401 beacuse we have 401 theats. including 1 for the bias unit that's why 1 extra 
%    X=[1 x1 x2 x3 x4 ......x400]  => 1st row 
%	   [1 x1 x2 x3 x4 ......x400]  => 2nd row
%      [1 x1 x2 x3 x4 ......x400]  => 3rd
%      [1 x1 x2 x3 x4 ......x400]  .....and so on till 5000
%
%	all_theta=[Q0 Q1 Q2 Q3.....Q400 ]  => this helps in computing class 1 
%			  [Q0 Q1 Q2 Q3.....Q400 ]  => this helps in computing class 2
%			  [Q0 Q1 Q2 Q3.....Q400 ]  => this helps in computing class 3
%			  [Q0 Q1 Q2 Q3.....Q400 ]  =>this helps in computing class 4......and so on till class 10
%  Now we want to see how much does an image corresoponds to a particular class 
%  i.e in terms of our matrix if i take any row(it represents an image with 20X20) which class does it corresponds to 
%  We pass our row to the Theats for class 1, class 2 , class3 and so on to identify which row it corresponds to.
%  we do all_theta' 
%  all_theta[Q0		Q0    
%            Q1		Q1
%            Q2		Q2
%			 Q3		Q3
%			 Q4		Q4
%			 .		.	
%			 .		.	
%			 .		.
%			 Q10   	Q10
%			Class1  class2 .......and so on 
% So now when do multiply X with all_theta' we are basically passing our every row to compute which class it belongs to and then we take the maximum of the row i.e with the 
% heighest probablity	 
%


computed_for_allclass=X*all_theta';
[temp,p]=max(computed_for_allclass,[],2);




% =========================================================================


end
