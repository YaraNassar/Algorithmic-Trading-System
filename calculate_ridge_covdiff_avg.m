function cov_diff_avg = calculate_ridge_covdiff_avg(test_actual_returns, test_factRet, training_returns,training_factRet,k)

%This function calculates the mean of the percentage difference of elements in the covariance matrix
% estimated by the factor loadings estimated through ridge regression, comparing these to actualized values in test period. 
% It runs ridge regression on every asset to obtain factor loadings, regressing the observations of asset returns in 
% training period with observations of factor returns in training period.

    % INPUTS: test_actual_returns (actual asset returns in the test period),
    % test_factRet (actual factor returns in test period), training_returns
    % (asset returns in training data), training_factRet (factor returns in
    % training data), k (ridge regression coefficient)
    % OUTPUTS: cov_diff_avg (arithmetic mean of the absolute value of 
    % percentage difference of elements in the covariance matrix compared
    % to realized values in test period.

    [~, p] = size(training_factRet); % number of factors
    n = size(test_actual_returns,2); % number of assets
    
   
    
    % Data matrix
    X = training_factRet;
    B = ones(p, n); % preallocate space

    
    for i = 1:n
    % perform ridge regression on each asset. The function returns factor
    % loadings for the p factors + the constant term (alpha)
        B(:,i) = ridge(training_returns(:,i), X(:,2:end), k, 0);
    end
    
    
    
    %% calculate mu and Q based on training data
    a = B(1,:)';
    V = B(2:end,:);

    
    [T, p] = size(test_factRet(:,2:end)); 
    % number of observations in test period and number of factors

    % calculate the residuals and diagonal matrix of residual error
    ep       = test_actual_returns - test_factRet * B;
    sigma_ep = 1/(T - p - 1) .* sum(ep .^2, 1);
    D        = diag(sigma_ep);
    
    
    % calculate the factor covariance matrix
    f_bar = mean(test_factRet(:,2:end),1)';
    F     = cov(test_factRet(:,2:end));
    
    
    
    % calculate the covariance predicted by regression for testing period
    Q_t  = V' * F * V + D;
        
    % Sometimes quadprog shows a warning if the covariance matrix is not
    % perfectly symmetric.
    Q_t = (Q_t + Q_t')/2;
    
    % calculate realized covariance Q in test period
   
    Q = cov(test_actual_returns);
    Q = (Q + Q')/2;

    % calculate the mean absolute value of the percentage difference in
    % covariance
    cov_diff_avg = mean(abs(Q - Q_t)./abs(Q), 'all');
end

