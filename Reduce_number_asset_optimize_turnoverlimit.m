function x = Reduce_number_asset_optimize_turnoverlimit(x_unrestricted, k, L, mu, Q, x0)

    % This function performs mean variance error tracking reduction to
    % determine a portfolio with reduced cardinality.

    % INPUTS:
    % x_unrestricted - the original portfolio weights of the unrestricted
    % portfolio
    % k - the cardinality limit
    % L - the initial cap on turnover
    % mu - estimate of expected return
    % Q - covariance matrix of assets
    % x0 - the previous portfolio weights (determined in previous
    % rebalancing)
    % OUTPUTS: x - the risk parity portfolio (unrestricted in cardinality)

    
    %----------------------------------------------------------------------

 

I = false; % track whether we are finished optimization
while I == false

    n = size(mu,1); % number of assets

    % If it is the first time building the portfolio, no limit on
    % turnover
    if 1 && ~any(x0)
        L = 2;
    end
    
    % resize mu and Q
    mu = [mu; zeros(2*n,1)]; 
    Q  = [Q zeros(n,2*n); zeros(2*n,3*n)];
    x_unrestricted_resized = [x_unrestricted' zeros(1,2*n)]';

    % equality constraints
    Aeq = [ones(1,n) zeros(1,2*n)];
    beq = 1;

    % inequality constraints
    A = [
        zeros(1,n) ones(1,n) zeros(1,n);
        eye(n) -1 * eye(n) zeros(n,n);
        eye(n) zeros(n,n), -eye(n);
        -eye(n) zeros(n,n) -eye(n);
        zeros(1, 2*n) ones(1,n);
        -mu'];

    b = [k; zeros(n,1); x0; -x0; L; -mu' * x_unrestricted_resized];
    
    
    lb = [zeros(1,3*n)];
    ub = [ones(1,3*n)];
    
    % define variable types
    varTypes = [repmat('C', n, 1) repmat('B', n, 1) repmat('C', n, 1) ];

    % Define variable names
    namesx = cellfun(@(c)['x_' c], string(1:1:n), 'uni', false); 
    namesy = cellfun(@(c)['y_' c], string(1:1:n), 'uni', false); 
    namesz = cellfun(@(c)['z_' c], string(1:1:n), 'uni', false); 
    % Combine both name vectors
    names = [namesx namesy namesz];

    %% Set up gurobi
    clear model;

    % Assign the variable names
    model.varnames = names;
    
    % Gurobi accepts an objective function of the following form:
    % f(x) = x' Q x + c' x 
    
    % Define the Q matrix in the objective 
    model.Q = sparse(Q);
    
    % define the c vector in the objective
    model.obj = -2 *x_unrestricted_resized' * Q;
    
    % Gurobi only accepts a single A matrix, with both inequality and equality
    % constraints
    model.A = [sparse(A); sparse(Aeq)];
    
    % Define the right-hand side vector b
    model.rhs = full([b; beq]);
    
    % Indicate whether the constraints are ">=", "<=", or "="
    model.sense = [ repmat('<', (3+3*n), 1) ; repmat('=', 1, 1) ];
    
    % Define the variable type (continuous, integer, or binary)
    model.vtype = varTypes;
    
    % Define the variable upper and lower bounds
    model.lb = lb;
    model.ub = ub;

    model.modelsense = 'min'; % maximization

    % Set some Gurobi parameters to limit the runtime and to avoid printing the
% output to the console. 
    clear params;
    params.TimeLimit = 100;
    params.OutputFlag = 0;
    
    try
        % check if solution is infeasible, which is possible given the
        % general constraint on the turnover and the minimum return
        % constraint. if infeasible, increase the turnover limit
        results = gurobi(model,params);

        x = results.x(1:n);
        I = true; % finished optimization
    catch
        % if we encounter an error, increase the turnover constraint up to
        % 0.4
        L = L + 0.05;
        if L > 0.4
            % if we exceeded this threshold
            x = x0; % use previous weights
            I = true; % finished optimization
        end
    end
end


end