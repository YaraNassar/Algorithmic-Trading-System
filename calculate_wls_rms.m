function RMS = calculate_wls_rms(test_actual_returns, test_factRet, training_returns,training_factRet,tau)

    % This function calculates the root mean square error of
    %returns compared to actualized values in test period. It runs weighted
    %least squares regression on every asset to obtain factor loadings, regressing the
    %observations of asset returns in training period with observations of
    %factor returns in training period.

    % INPUTS: test_actual_returns (actual asset returns in the test period),
    % test_factRet (actual factor returns in test period), training_returns
    % (asset returns in training data), training_factRet (factor returns in
    % training data), tau (the parameter for the exponential decay weight term)
    % OUTPUTS: RMS (root mean square error of the returns compared to actualized values)
%----------------------------------------------------------------------

% Number of observations and factors
    [T, n] = size(training_returns);  % length of training data and number of assets
   
    
    % Regression coefficients
    
  
    w = exp(1).^(tau.*(T:-1:1))'; % the weight of the observations in training data. 
    % Each observation is weighted by exp(tau * number of observations in
    % the past). For example, the most recent observation is weighted by
    % exp(tau * 1)
    
    
    for i = 1:n
        % perform WLS on each asset. Note that the factor return matrix already contains a column of ones. 
        % The function returns factor loadings for the p factors + the constant term (alpha)
        B(:,i) = lscov(training_factRet, training_returns(:,i), w);
    end

    N = size(test_factRet,1); % number of observations in test data
    n = size(test_actual_returns,2); % number of assets
    
    %% calculate RMS

    % compute the predicted return for each test value given the factor
    % loadings

    predicted_returns = test_factRet*B;
    
    % calculate the root mean square error for all observations in test
    % period by comparing with the actual returns
    diff = test_actual_returns - predicted_returns;
    RMS = sqrt(sum(diff.^2, "all")/(N*n));
end