function x = Reduce_number_asset_optimize(x_unrestricted, k, mu, Q)

% This function takes weight x_unrestricted

% Returns weights x for a portfolio with k maximum number of assets.

    % Since our problem now has 2n variables, we must re-size mu and Q 
% accordingly.

    n = size(mu,1); % number of assets

    mu = [mu; zeros(n,1)]; 
    Q  = [Q zeros(n); zeros(n,2*n)];
    x_unrestricted_resized = [x_unrestricted' zeros(1,n)]';


    Aeq = [ones(1,n) zeros(1,n)];
    beq = 1;

    A = [
        zeros(1,n) ones(1,n);
        eye(n) -1 * eye(n);
        -mu'];

    b = [k; zeros(n,1); -mu' * x_unrestricted_resized];

    lb = [zeros(1,2*n)];
    ub = [ones(1,2*n)];

    varTypes = [repmat('C', n, 1) repmat('B', n, 1)];

    % Define variable names
    namesx = cellfun(@(c)['x_' c], string(1:1:n), 'uni', false); 
    namesy = cellfun(@(c)['y_' c], string(1:1:n), 'uni', false); 
    
    % Combine both name vectors
    names = [namesx namesy];

    %% Set up gurobi
    clear model;

    % Assign the variable names
    model.varnames = names;
    
    % Gurobi accepts an objective function of the following form:
    % f(x) = x' Q x + c' x 
    
    % Define the Q matrix in the objective 
    model.Q = sparse(Q);
    
    % define the c vector in the objective (which is a vector of zeros since
    % there is no linear term in our objective)
    model.obj = -2 *x_unrestricted_resized' * Q;
    
    % Gurobi only accepts a single A matrix, with both inequality and equality
    % constraints
    model.A = [sparse(A); sparse(Aeq)];
    
    % Define the right-hand side vector b
    model.rhs = full([b; beq]);
    
    % Indicate whether the constraints are ">=", "<=", or "="
    model.sense = [ repmat('<', (2+n), 1) ; repmat('=', 1, 1) ];
    
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
    
    results = gurobi(model,params);

    x = results.x(1:n);
end