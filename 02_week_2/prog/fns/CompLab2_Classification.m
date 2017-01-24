function [] = CompLab2_Classification(Polynomial_Order)

% A small constant used to avoid log of zero problems
SMALL_NOS = 1e-200;

% Set the prior precision
alpha=100; 

% Load and prepare train & test data
X       = load('Data_Train.txt');
Xt      = load('Data_Test.txt');
t       = X(:,3);
X(:,3)  = [];
tt      = Xt(:,3);
Xt(:,3) = [];

% Plot two classes in train set
figure(1)
plot(X(find(t==1),1),X(find(t==1),2),'r.');
hold
plot(X(find(t==0),1),X(find(t==0),2),'o');
title('Scatter Plot of Data from Classes');
disp('Two overlapping Non-Gaussian Class Distributions')
disp(' ')

% Create Polynomial Basis
XX = []; XXt = [];
for i = 0:Polynomial_Order
    XX = [XX X.^i];
    XXt = [XXt Xt.^i];
end
[N,D] = size(XX);
Nt    = size(XXt,1);

% Newton routine to find MAP values of w
% Fix number of steps to 10 and initial estimate to w=0
N_Steps = 10;
w = zeros(D,1);

disp('Optimising using Newton steps:')
for m=1:N_Steps    
    % Newton Step
    P = 1./(1 + exp(-XX*w));
    A = diag(P.*(1-P));
    H = inv(XX'*A*XX + eye(D)./alpha);
    w = H*XX'*(A*XX*w + t - P);

    % Compute new likelihood and unormalised posterior values for training
    % data
    f             = XX*w; % training data
    LogPrior      = log(gauss(zeros(1,D),eye(D).*alpha,w'));
    LogLikelihood = f'*t - sum(log(1+exp(f))); % training likelihood
    ljt           = LogLikelihood + LogPrior;
    fprintf('Log-Likelihood = %f, Joint-Likelihood = %f\n',LogLikelihood,ljt)
end

disp(' ')
disp(['Final parameters are: '])
disp(w)

% Compute Overall performance
Train_Like = LogLikelihood;
ft         = XXt*w; % test data
Test_Like  = ft'*tt - sum(log(1+exp(ft))); % test likelihood

Train_Error = 100 - 100*sum( (1./(1+exp(-XX*w)) > 0.5) == t)/N; % number of miss-classifications
Test_Error  = 100 - 100*sum( (1./(1+exp(-XXt*w)) > 0.5) == tt)/Nt;

fprintf('\n\nClassifier Performance Statistics using MAP Value\n');
fprintf('Training Likelihood = %f, Training 0-1 Error = %f\n',Train_Like,Train_Error);
fprintf('Test Likelihood = %f, Test 0-1 Error = %f\n',Test_Like,Test_Error);




end
