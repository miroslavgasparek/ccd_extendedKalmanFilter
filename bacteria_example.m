%% 19 May 2019 Miroslav Gasparek
% Discrete-continuous constrained Kalman filter for the bacteria-waste system
% as described in
%?Lewis, F. L., Xie, L., Popa, D., Xie, L., & Popa, D. (2017). 
% Optimal and Robust Estimation. CRC Press. 
% https://doi.org/10.1201/9781315221656

% The system of equatiosn is originally taken from: 
% Luenberger, D. G. (1979). Introduction to Dynamic Systems: 
% Theory, Models, and Applications. John Wiley & Sons.

% The equations describe the dynamics of the industrial plant, where
% the waste is fed into a pool of bacteria that transforms the waste 
% into non-polluting forms. If the bacteria concentration is [B] and
% pollutant concentration is [P] then the continuous state-space model 
% is described as: 
% 
% d[B]/dt = a*[B]*[P]/([P] + K) + w1
% d[P]/dt = - b* [B]*[P]/([P] + K) + w2
% 
% The discrete observations have the form of:
% z(k) = [P] + c*[B]^2 + v(k)
% 
% where a, b, c, K are constants, w1, w2 are uncorrelated
% Gaussian process noises with covariances q1 and q2, and v(k) is the 
% Gaussian measurement noise with covariance R 

clear; clc; close all;
 
%% Define the noise parameters
n = 2;      %number of states

G = [1, 0; 
     0, 1];  % Process noise matrix

R = 0.02;      % Covariance of measurement

q1 = 0.02;
q2 = 0.01;

Q = [q1, 0;
     0,  q2];   % Covariances of process

%% Initial conditions and system parameters
% System parameters
a = 1.2;
b = 0.4;
Km = 0.8;
c = 0.5;

% Initial conditions for the states
x_init(1) = 0.8;
x_init(2) = 0.4;

% Initial conditions for the cov. matrix elements
Pmat_init = [0.1, 0.1, 0.1, 0.1];
    
% px(1:4)=cov. mat. elements
% px(5:6)= state estimates
px_init = [Pmat_init, x_init];

%% Simulation and sampling parameters
Ts = 0.05; % Sampling period is 0.1x the measurement update
Tmeas = 0.5; % Measurement update
Tspan = [0 50]; % Timespan of the simulation
Tsim = Tspan(2); % End of the simulation

%% Define the function of the state
x = sym("x",[(n+1)*n, 1]);
f_state = [(a*x(5)*x(6))/(x(6) + Km );
           -(b*x(5)*x(6))/(x(6) + Km )];
       
%% Define the function for the observation
h_obs = x(6) + c*x(5)^2;

%% Insert the constraints
% The constraints should be of the form:
% A_ineq * x <= b_ineq (inequality constraint)

% In this case, the inequality constraint below restricts the concentrations 
% so that they are non-negative, i. e. x(5) >= 0; x(6) >= 0;
A_ineq = [-1, 0;
           0, -1];

b_ineq = zeros(n,1);

%% %% Running the Kalman Filter for the state estimation %% %%

[t_vec, Tmeas_vec, Pvec, Psys, zvec, ms_error] = dc_ekf(n, f_state, h_obs, Ts, Tmeas, Tspan, Tsim, px_init, x_init, G, R, Q, A_ineq, b_ineq);


%% Plot the covariance matrix elements
figure(1);
sgtitle('Elements of the covariance matrix','fontsize',20,'interpreter', 'latex')
hold on
subplot(4,1,1); plot(t_vec,Pvec(:,1),'k.'); ylabel('$p_{1}$','fontsize',15,'interpreter', 'latex'); grid on;
xlabel('Time (h)','fontsize',15,'interpreter', 'latex');

subplot(4,1,2); plot(t_vec,Pvec(:,2),'k.'); ylabel('$p_{2}$','fontsize',15,'interpreter', 'latex'); grid on;
xlabel('Time (h)','fontsize',15,'interpreter', 'latex');

subplot(4,1,3); plot(t_vec,Pvec(:,3),'k.'); ylabel('$p_{3}$','fontsize',15,'interpreter', 'latex'); grid on;
xlabel('Time (h)','fontsize',15,'interpreter', 'latex');

subplot(4,1,4); plot(t_vec,Pvec(:,4),'k.'); ylabel('$p_{4}$','fontsize',15,'interpreter', 'latex'); grid on;
xlabel('Time (h)','fontsize',15,'interpreter', 'latex');

fig = gcf;
fig.Position = [136 169 864 629];

%% Plot the observation and the state estimates + states
figure(2);
sgtitle('Kalman filter estimation of the bacteria and pollutant concentration','fontsize',20,'interpreter', 'latex')
hold on
% Plot the observation
subplot(3,1,1); plot(Tmeas_vec, zvec(:,1), 'b.', 'MarkerSize', 15); ylabel('z'); grid on;
title('Noisy measurements vs. time','fontsize',15,'interpreter', 'latex')
ylabel('Observation (AU)','fontsize',15,'interpreter', 'latex')
xlabel('Time (h)','fontsize',15,'interpreter', 'latex');
ax = gca;
ax.YLim = [0, 2.5];

% Plot the bacterial concentration estimate
subplot(3,1,2); 
hold on
plot(t_vec,Pvec(:,5),'r.');
plot(t_vec,Psys(:,1),'k--','LineWidth',1.5);
title('Bacterial concentration vs. time','fontsize',15,'interpreter', 'latex')
ylabel('[B] (AU)','fontsize',15,'interpreter', 'latex'); grid on;
xlabel('Time (h)','fontsize',15,'interpreter', 'latex');
legend('estimated [B]', 'true [B]','interpreter','latex','location','best','fontsize', 15)
ax = gca;
ax.YLim = [0, 2.5];

% Plot the estimate of the amount of the waste
subplot(3,1,3); 
hold on
plot(t_vec,Pvec(:,6),'r.');
plot(t_vec,Psys(:,2),'k--','LineWidth',1.5);
title('Pollutant concentration vs. time','fontsize',15,'interpreter', 'latex')
ylabel('[P] (AU)','fontsize',15,'interpreter', 'latex'); grid on;
xlabel('Time (h)','fontsize',15,'interpreter', 'latex');
legend('estimated [P]', 'true [P]','interpreter','latex','location','best','fontsize', 15)
ax = gca;
ax.YLim = [0, 0.6];

fig = gcf;
fig.Position = [136 169 864 629];

