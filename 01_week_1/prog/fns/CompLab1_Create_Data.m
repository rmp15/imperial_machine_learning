function [x, y] = CompLab1_Create_Data()

% Standard deviation of the noise to be added
Noise_std = 100;

% Set up the input variable, 100 points between -5 and 5
x = (-5:10/(100-1):5)';

% Calculate the true function and add some noise
y = 5*x.^3 - x.^2 + x + Noise_std*randn(size(x));


end