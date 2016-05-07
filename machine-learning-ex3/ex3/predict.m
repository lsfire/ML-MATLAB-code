function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X = [ones(m, 1) X];
hidden = sigmoid((Theta1*X'));%25 * 5000
m2 = size(hidden,2); % 5000
hidden = [ones(m2, 1) hidden'];% 5000 * 26
result = sigmoid((Theta2*hidden'));% 10 * 5000

for i=1:m
    temp = result(:,i);
    max_value = max(temp);
    index = find(temp == max_value);
    p(i) = mod(index,10);
end
% =========================================================================


end
