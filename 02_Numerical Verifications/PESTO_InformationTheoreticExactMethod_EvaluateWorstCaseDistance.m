clear all; clc;
% In this example, we use the Information-Theoretic Exact Method for solving the 
% L-smooth m-strongly convex minimization problem
%   min_x F(x); 
%   for notational convenience we denote xs=argmin_x F(x).
%
% We show how to compute the worst-case value of ||z_N-xs|| when zN is
% obtained by doing N steps of the method starting with an initial
% iterate satisfying ||z0-xs||<=1.
%
% [1] 

% (0) Initialize an empty PEP
P = pep();

L = 1; m = .1; kappa = m/L;
% (1) Set up the objective function
param.L  = L;               % Smoothness parameter
param.mu = m;               % Strong convexity parameter

% F is the objective function
F = P.DeclareFunction('SmoothStronglyConvex',param); 

% (2) Set up the starting point and initial condition
x0      = P.StartingPoint();         % x0 is some starting point
[xs,fs] = F.OptimalPoint();          % xs is an optimal point, and fs=F(xs)
P.InitialCondition( (x0-xs)^2 <= 1); % Initial condition ||x0-xs||^2<= 1

% (3) Algorithm
N = 3;		% number of iterations

% Algorithmic parameters
Akn = @(x) ( ( (1+kappa)*x+2*(1+sqrt((1+x)*(1+kappa*x)) ))/(1-kappa)^2); 
Akv = zeros(N+1,1);
for i = 1:N
    Akv(i+1) = Akn(Akv(i));
end
Ak  	= @(k)(Akv(k+1));
betak  	= @(k)(Ak(k)/(1-kappa)/Ak(k+1));
deltak 	= @(k)(1/2 * ( (1-kappa)^2*Ak(k+1) -(1+kappa)*Ak(k))/(1+kappa+kappa*Ak(k)) );

% Iterates
y = cell(N+1,1);% store iterates in a cell
x = cell(N+1,1);% store iterates in a cell
z = cell(N+1,1);% store iterates in a cell
g = cell(N+1,1);% store iterates in a cell
f = cell(N+1,1);% store iterates in a cell

x{1}    = x0;
z{1}    = x0;

% Algorithm
for i=0:N-1
    y{i+1}  = betak(i) * x{i+1} + (1-betak(i)) * z{i+1};
    [g{i+1},f{i+1}] = F.oracle(y{i+1});
    x{i+2}  = y{i+1} - 1/L * g{i+1};
    z{i+2}  = (1-kappa*deltak(i))*z{i+1} + kappa*deltak(i) * y{i+1} - deltak(i)/L * g{i+1};
end

% (4) Set up the performance measure
obj = ( z{N+1} - xs)^2;
P.PerformanceMetric(obj); % Worst-case evaluated for ||z_N-xs||^2

% (5) Solve the PEP
P.solve()

% (6) Evaluate the output, and verify that it is equal to
%       1/(1+kappa*Ak(N))
% and that it satisfies the inequality
%  1/(1+kappa*Ak(N)) <= (1-sqrt(kappa))^(2*N)/((1-sqrt(kappa))^(2*N)+kappa)
% 
[double(obj) 1/(1+kappa*Ak(N))  (1-sqrt(kappa))^(2*N)/((1-sqrt(kappa))^(2*N)+kappa)]
    







