function [X_norm , mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

% [n , m] = size(X);
% for i = 1 : m
% 	mu(i:i) = mean(X (:, i:i));
% 	sigma(i:i) = std(X (:, i:i));
% 	X_norm (:, i:i) = (X(:, i:i) - mu (i:i))./sigma(i:i);
% 	%fprintf(' x = [%.5f] \n', [((X(:, i:i) - mu (i:i))/sigma(i:i))(1:10)]');
% 	%fprintf('First 10 examples from the dataset: \n');
% 	%fprintf(' x = [%.0f %.0f] \n', [X_norm(1:10,:)]');
% end

[n , m] = size(X);
for i = 1 : m
	mu(i:i) = mean(X (:, i:i));
	sigma(i:i) = std(X (:, i:i));
	X_norm (:, i:i) = (X(:, i:i) - mu (i:i))./sigma(i:i);
	%fprintf(' x = [%.5f] \n', [((X(:, i:i) - mu (i:i))/sigma(i:i))(1:10)]');
	%fprintf('First 10 examples from the dataset: \n');
	%fprintf(' x = [%.0f %.0f] \n', [X_norm(1:10,:)]');
end

% ============================================================
end
