function x = Project2_Function(periodReturns, periodFactRet, x0)

    % Use this function to implement your algorithmic asset management
    % strategy. You can modify this function, but you must keep the inputs
    % and outputs consistent.
    %
    % INPUTS: periodReturns, periodFactRet, x0 (current portfolio weights)
    % OUTPUTS: x (optimal portfolio)
    %
    % An example of an MVO implementation with OLS regression is given
    % below. Please be sure to include comments in your code.
    %
    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    n = size(periodReturns,2); %number of assets
    
    % Regression to obtain factor loading estimates
    N = 35; % length of training data
    T = 6; % length of test data for determination of regression method;
    from_start = false; % whether we take training data from the start of available data for regression parameter estimation
    [mu, Q] = return_cov_estimator(periodReturns, periodFactRet, T, N, from_start);
    
    % Risk parity optimization
    c = 10; % constant in convex reformulation
    x_unrestricted = Risk_Parity_convex(Q,c); % the weights unrestricted for cardinality

    k = max(n-10,10); % cardinality limit 
    L = 0.20; % initial turnoverlimit

    x = Reduce_number_asset_optimize_turnoverlimit(x_unrestricted, k, L, mu, Q, x0);

    
    %----------------------------------------------------------------------
end
