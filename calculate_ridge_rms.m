function RMS = calculate_ridge_rms(test_actual_returns, test_factRet, training_returns,training_factRet,k)
%calculate_ridge_rms This function calculates the root mean square error of
%returns compared to actualized values in test period. It runs ridge
%regression on every asset to obtain factor loadings, regressing the
%observations of asset returns in training period with observations of
%factor returns in training period.

    % INPUTS: test_actual_returns (actual asset returns in the test period),
    % test_factRet (actual factor returns in test period), training_returns
    % (asset returns in training data), training_factRet (factor returns in
    % training data), k (ridge regression coefficient)
    % OUTPUTS: RMS (root mean square error of the returns compared to actualized values)


    [~, p] = size(training_factRet); % number of factors + 1
    n = size(test_actual_returns,2); %number of assets
    N = size(test_factRet,1); % number of observations in test period
   
    
    % Data matrix
    X = training_factRet;
    B = ones(p, n); % allocate space to store factor loadings

    
    for i = 1:n
    % perform ridge regression on each asset. The function returns factor
    % loadings for the p factors + the constant term (alpha)
        B(:,i) = ridge(training_returns(:,i), X(:,2:end), k, 0); 
    end
    
    
    
    %% calculate RMS

    % compute the predicted return for each test value given the factor
    % loadings

    predicted_returns = test_factRet*B;
    
    % calculate the root mean square error for all observations in test
    % period by comparing with the actual returns
    diff = test_actual_returns - predicted_returns;
    RMS = sqrt(sum(diff.^2, "all")/(N*n));
end

