function [mu, Q] = return_cov_estimator(periodReturns, periodFactRet, T, N, from_start)

    % This function calculates the best estimate of mu and Q by
    % perform a regression method to obtain loadings for the factors in
    % periodFactRet. The function will determine an appropriate parameter
    % for the regression method and select the regression method based on
    % the error metrics presented in the paper. mu and Q will be estimated
    % using the factor loadings.

    % Estimation of mu and Q will be conducted separately, with different
    % regression.

    % INPUTS: periodReturns, periodFactRet, x0 (current portfolio weights),
    % T (length of test data), from_start (true if we take data from the
    % start as training data; false if we use sliding window method with
    % length determined by governed by N
    % N (length of training data, only relevant
    % when from_start = false)
    % OUTPUTS: mu, Q (our best estimation for these values)

    
    %----------------------------------------------------------------------

    % Define test data
    test_actual_returns = periodReturns(end-T:end,:);
    test_factRet = [ones(size(periodFactRet(end-T:end,:),1),1) periodFactRet(end-T:end,:)];
    
    % define training data. Ensure that the training data is separate from
    % the test data.
    if from_start == true  || N >= (length(periodReturns) - T)
        % here training data starts from the start of period returns
        training_returns = periodReturns(1:end-T-1,:);
        training_factRet = [ones(length(periodFactRet(1:end-T-1,:)), 1) periodFactRet(1:end-T-1,:)];

    else 
        % sliding window will be used
        training_returns = periodReturns(end-T-1-N:end-T-1,:);
        training_factRet = [ones(length(periodFactRet(end-T-1-N:end-T-1,:)),1)  periodFactRet(end-T-1-N:end-T-1,:)];
    end
    
    %% Estimate mu 

    % First determine the ideal method and parameter for each method to run
    % regression.

    % Ridge

    % bounds to search for k - determined through testing
    k_min = 0.5;
    k_max = 150;
    
    % calculate the k that minimizes RMS comparing the predicted data with
    % the test data
    k_optimal = fminbnd(@(k)calculate_ridge_rms(test_actual_returns, test_factRet, training_returns,training_factRet,k),k_min,k_max);
    
    % determine the RMS at this k_optimal
    ridge_RMS_optimal = calculate_ridge_rms(test_actual_returns, test_factRet, training_returns,training_factRet,k_optimal);

    % WLS 
    
    % bounds to search for tau - determined through testing
    tau_min = -0.1;
    tau_max = -0.001;

    % calculate the k that minimizes RMS comparing the predicted data with
    % the test data
    tau_optimal = fminbnd(@(k)calculate_wls_rms(test_actual_returns, test_factRet, training_returns,training_factRet,k),tau_min,tau_max);
    
    % calculate optimal RMS for this value of tau
    wls_RMS_optimal = calculate_wls_rms(test_actual_returns, test_factRet, training_returns,training_factRet,tau_optimal);

    % Now that the parameter and method has been determined, we will
    % perform the actual regression. 
    % 
    
    
    % set the data to regress depending on our values of from_start and N
    if from_start == true  || N >= (length(periodReturns) - T)
        % use data from start
        returns_fitting = periodReturns;
        factorRet_fitting = periodFactRet;
    else
        % use data going back N observations
        returns_fitting = periodReturns(end-N:end,:);
        factorRet_fitting = periodFactRet(end-N:end,:);
    end
% 
    returns_fitting = periodReturns;
    factorRet_fitting = periodFactRet;

    % compare the RMS between the two methods
%     if ridge_RMS_optimal <= wls_RMS_optimal
%         % use ridge if its RMS is lower to estimate mu
%         [mu, ~] = Ridge(returns_fitting,factorRet_fitting, k_optimal);
%         
%     else
%         % otherwise, use WLS to estimate mu
%         [mu, ~] = weighted_OLS(returns_fitting,factorRet_fitting, tau_optimal);
%         
%     end
    
    % From testing, we only use ridge for this project.
    [mu, ~] = Ridge(returns_fitting,factorRet_fitting, k_optimal);
    
    %% Estimate covariance
    % Ridge

    % These bounds were determined through testing
    k_min = 10;
    k_max = 10000;
    
    % find the value of k such that the average absolute percent difference
    % in covariance estimates is minimized
    k_optimal = fminbnd(@(k)calculate_ridge_covdiff_avg(test_actual_returns, test_factRet, training_returns,training_factRet,k),k_min,k_max);
    
    % perform the regression and estimate Q
    [~, Q] = Ridge(returns_fitting,factorRet_fitting, k_optimal);
    
    %----------------------------------------------------------------------
end
