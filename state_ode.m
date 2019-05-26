%% 16 May 2019 Miroslav Gasparek
% ODE describing the time evolution of the covariance matrix terms and 
% states for the extended continuous-discrete Kalman Filter
function system_ode = state_ode(f_state, n)

x = sym("x",[n*(n+1), 1]);
syms t
system_ode = matlabFunction(f_state,'vars', {t, x(n*n+1:n*(n+1))},'file','system_ode');
end

