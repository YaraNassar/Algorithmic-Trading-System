function [mu, Q] = Ridge(returns, factRet,k)
% This function calculates mean return and return covariance matrix by running
% ridge regression, regressing the observations of asset returns on observations of
% factor returns.

    % INPUTS: returns (asset returns), factRet (factor returns), k (ridge regression coefficient)
    % OUTPUTS: mu, Q
    

    [T, p] = size(factRet); % number of observations and factors
    n = size(returns,2); % number of assets
    
    % Data matrix
    X = [ones(T,1) factRet]; % introduce column of ones
    B = ones(p+1, n); % preallocate storage
% Regression coefficients

    
    for i = 1:n
    % perform ridge regression on each asset. The function returns factor
    % loadings for the p factors + the constant term (alpha)
        B(:,i) = ridge(returns(:,i), X(:,2:end), k, 0);
    end

    % Separate B into alpha and betas
    a = B(1,:)';     
    V = B(2:end,:); 
    
    % Residual variance
    ep       = returns - X * B;
    sigma_ep = 1/(T - p - 1) .* sum(ep .^2, 1);
    D        = diag(sigma_ep);
    
    % Factor expected returns and covariance matrix
    f_bar = mean(factRet,1)';
    F     = cov(factRet);
    
    % Calculate the asset expected returns and covariance matrix
    mu = a + V' * f_bar;
    Q  = V' * F * V + D;
    
    % Sometimes quadprog shows a warning if the covariance matrix is not
    % perfectly symmetric.
    Q = (Q + Q')/2;
end

