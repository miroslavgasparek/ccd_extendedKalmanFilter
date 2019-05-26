%% 10 May 2019 Miroslav Gasparek
% ODE describing the time evolution of the covariance matrix terms and 
% states for the extended continuous-discrete Kalman Filter
function DiffEqn = ode_px(f_state, n, G, Q)

x = sym("x",[(n+1)*n, 1]);
syms t
Pmat = reshape(x(1:n*n).',n,n).';

fun_sym = matlabFunction(f_state);
Amat = jacobian(fun_sym, x( (n*n+1): n*(n+1) ));
Pdot = Amat*Pmat + Pmat*Amat' + G*Q*G';
Peqn_full = [reshape(Pdot, [n^2,1]); f_state];

DiffEqn = matlabFunction(Peqn_full,'vars', {t, x},'file','ode_ekf');
end

