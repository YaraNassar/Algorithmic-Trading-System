function  [mu, Q] = weighted_OLS(returns, factRet, tau)
% This function calculates mean return and return covariance matrix by running
% weighted least squares regression, regressing the observations of asset returns on observations of
% factor returns. The observations will be weighted by exp(tau * number of
% observations in the past)

    % INPUTS: returns (asset returns), factRet (factor returns), tau (decay parameter)
    % OUTPUTS: mu, Q
 
   
    %----------------------------------------------------------------------
    
    % Number of observations and factors
    [T, p] = size(factRet); % number of observations and factors
    [T,n] = size(returns); % number of assets
    
    % Data matrix
    X = [ones(T,1) factRet];
    
    % the weight of the observations in training data. 
    % Each observation is weighted by exp(tau * number of observations in
    % the past). For example, the most recent observation is weighted by
    % exp(tau * 1)
    
    w = exp(1).^(tau.*(T:-1:1))';

    
    for i = 1:n
        % perform WLS on each asset. 
        % The function returns factor loadings for the p factors + the constant term (alpha)
        B(:,i) = lscov(X, returns(:,i), w);
    end
    % Separate B into alpha and betas
    a = B(1,:)';     
    V = B(2:end,:); 
    
    % Residual variance calculation
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
    
    %----------------------------------------------------------------------
    
end