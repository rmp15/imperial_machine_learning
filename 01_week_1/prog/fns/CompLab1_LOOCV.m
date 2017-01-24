function [cv_err, cv_std] = CompLab1_LOOCV(x, y, PolyOrder)

% x and y are column vectors containing the inputs and targets respectively
% PolyOrder is the order of polynomial model used for fitting

NumOfDataPairs = length(x);

% First construct design matrix of given order
X = ones(NumOfDataPairs, PolyOrder+1);
for i = 1:PolyOrder
    X(:,i+1) = x.^i;
end

% Initialise CV variable for storing results
CV = [];

for n = 1:NumOfDataPairs
    
    % Create testing design matrix and target data, leaving one out each time
    Test_X = X([1:n-1 n+1:NumOfDataPairs], :);
    Test_y = y([1:n-1 n+1:NumOfDataPairs]);
    
    % Create training design matrix and target data
    Train_X = X(n, :);
    Train_y = y(n);
    
    % Learn the optimal paramerers using MSE loss
    Paras_hat = (Test_X'*Test_X)\Test_X'*Test_y;
    Pred_y    = Train_X*Paras_hat;
    
    % Calculate the MSE of prediction using training data
    CV        = [CV; (Pred_y - Train_y).^2];
end

% Average the results
cv_err = mean(CV);
cv_std = sqrt(((NumOfDataPairs-1)/NumOfDataPairs)*sum((CV - cv_err).^2));

end