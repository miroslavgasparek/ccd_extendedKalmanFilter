%% 19 May 2019 Miroslav Gasparke
% Discrete-Continous Extented Kalman filter for the state estimation
% in the continuous systems with the discrete-time measurements.

% The system is described by the equations
% dx/dt = f(x) + G * w
% z = h(x) + v

% Where w ~ (0, Q), v ~ (0, R) are process and measurement noise
% with covariance matrices Q and R 

%%% Inputs:

%%% Outputs:

%% %% Running the Kalman Filter for the state estimation %% %%

function [t_vec, Tmeas_vec, Pvec, Psys, zvec, ms_error] = dc_ekf(n, f_state, h_obs, Ts, Tmeas, Tspan, Tsim, px, x_init, G, R, Q, A_ineq, b_ineq)


% Auxilliary variables for the plotting
i = 0; % Index counter for the sampling
j = 0; % Index counter for the measurements

t_vec = 0:Ts:Tsim;           % Time vector, used for plotting
t_len = size(0:Ts:Tsim,2);  % Length of the time vector
px_len = size(px,2);        % Length of the covariance matrix and states vector
Pvec = zeros(t_len, px_len); % covariance matrix elements and states vector

% Define the symbolic variable
x = sym("x",[(n+1)*n, 1]);

% ODE without the noise for the comparison
system_ode = state_ode(f_state,n);
% Solve the ODE without the noise
sol_sys = ode45(@(t, x) system_ode(t, x), Tspan, x_init); 
% Evaluate the solution
Psys = deval(sol_sys, t_vec);
Psys = Psys';

% ODE for the cov. mat and state evolution
DiffEqn = ode_px(f_state, n, G, Q);

% Jacobian for the observation matrix
Hmat = jacobian(h_obs, x(n*n+1 : n*(n+1)));
Hmat = matlabFunction(Hmat, 'vars', {x.'});

% Matlab function handle for the observation function
h_fun = matlabFunction(h_obs, 'vars', {x.'});

    for t = 0:Ts:Tsim
        
        % Index update
        j = j+1;
        % Time update
        t1 = t + Ts; % Define the end time interval
        [td, px] = ode45(@(t, px) ode_ekf(t, px), [t, t1], px); % Solve the ODEs
        px = px(end,:);
        
        % Create the covariance matrix
        Pold = reshape(px(1:n*n).', [n,n]).';
        
        %% Measurement update
        if (mod(t1, Tmeas)== 0 )
            i = i+1;
            % Observation estimate
            H = Hmat(px);
%             H = H(any(H,2),:);
            delta = H*Pold*H' + R/Ts;
            % Kalman Gain
            K = (Pold*H')/delta;
            % Update the covariance matrix
            Pnew = (eye(n) - K*H)*Pold;
            px(1:n*n) = reshape(Pnew, [1,n*n]);
            
            % The (noisy) measurement zk, formed by noise addition to h(x)
            zk = h_fun(px) + (R*randn(1, size(h_obs,2) ));

            % State estimate after measurement update
            % px(n*n+1 : n*(n+1)) always describes the states
            px(n*n+1 : n*(n+1)) = px(n*n+1 : n*(n+1)) + (K*(zk - h_fun(px))')';
            
            %% Constraints check
            % Check if inequality constraints are present
            if (isempty(A_ineq) ~= 1 && isempty(b_ineq) ~= 1)
                if any(A_ineq*px(n*n+1 : n*(n+1))' > b_ineq)
                    px(n*n+1 : n*(n+1)) = quadprog(Pnew, -Pnew * px(n*n+1 : n*(n+1))', A_ineq, b_ineq);
                end
            end
                
            %%
            
            % Store the new observation in the vector of observations
            zvec(i,:) = zk;
            
            % Store the time at which the observation was realized
            Tmeas_vec(i) = t1;

        end
        
        % Store the vector of covariances and state estimates
        Pvec(j,:) = px(:);
    
    end

    % Calcuate the mean squared error between the state estimates and true states
    ms_error = goodnessOfFit(Pvec(:, n*n+1 : (n*(n+1)) ), Psys,'MSE');

end

