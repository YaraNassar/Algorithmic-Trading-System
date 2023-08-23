function x = Risk_Parity_convex(Q,c)
    % This function performs risk parity optimization on assets with
    % covariance matrix Q. The initial constant c is a positive real
    % number.

    % INPUTS:
    % Q - covariance matrix of assets
    % c - positive constant
    % OUTPUTS: x - the risk parity portfolio (unrestricted in cardinality)

    
    %----------------------------------------------------------------------
    n = size(Q,1); % number of assets
    

    
    fun1 =  @(y) 1/2 * y' * Q* y - c*sum(log(y)) ;
    x0 = 1/n.*(ones(n,1));
    
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = zeros(1,n);
    ub = Inf(1,n);
    options = optimoptions('fmincon', 'MaxFunctionEvaluations', 15000);
    
    y = fmincon(fun1,x0,A,b,Aeq,beq,lb, ub, [],options); % obtain optimal weights

    x = y ./ (sum(y)); % normalize weights
   
    

end