function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

c_list = [0.01, 003, 0.1, 0.3, 1, 3, 10, 30]
sigma_list = [0.01, 003, 0.1, 0.3, 1, 3, 10, 30]
min_mean = 2.^31

for _c=1:size(c_list, 2),
	for _sigma=1:size(sigma_list, 2),
		model= svmTrain(X, y, c_list(_c), @(x1, x2) gaussianKernel(x1, x2, sigma_list(_sigma)));
		predictions = svmPredict(model, Xval)
                _mean = mean(double(predictions ~= yval))
		if _mean < min_mean
			min_mean = _mean
			C = c_list(_c)
			sigma = sigma_list(_sigma)
		end
		
        end
end





% =========================================================================

end
