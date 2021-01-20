clear all; clc;
% In this example, we use the Information-Theoretic Exact Method for solving the 
% L-smooth m-strongly convex minimization problem
%   min_x F(x); 
%   for notational convenience we denote xs=argmin_x F(x).
%
% We show how to compute the worst-case value of 
% (f(y_N)-f_*-1/L/2 || f'(y_N) ||^2-m/2/(1-m/L)*||y{N+1}-1/L*g{N+1}-xs||^2
% when y_N is obtained by doing N steps of the method starting with an
% initial iterate satisfying ||z0-xs||<=1.
%

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
for i = 1:N+1
    Akv(i+1) = Akn(Akv(i));
end
Ak  	= @(k)(Akv(k+1));
betak  	= @(k)(Ak(k)/(1-kappa)/Ak(k+1));
etak 	= @(k)(1/2 * ( (1-kappa)^2*Ak(k+1) -(1+kappa)*Ak(k))/(1+kappa+kappa*Ak(k)) );

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
    z{i+2}  = z{i+1} + kappa*etak(i) * ( y{i+1} - z{i+1}) - etak(i)/L * g{i+1};
end
i = N;
y{i+1}  = betak(i) * x{i+1} + (1-betak(i)) * z{i+1};
[g{i+1},f{i+1}] = F.oracle(y{i+1});

% (4) Set up the performance measure
psi = (f{N+1}-fs-1/L/2*(g{N+1})^2-m/2/(1-m/L)*(y{N+1}-1/L*g{N+1}-xs)^2);
P.PerformanceMetric(psi); 

% (5) Solve the PEP
P.solve()

% (6) Evaluate the output and verify that it matches L/((1-kappa)*Ak(N+1))
% and that it is upper bounded by 
% L/(1-kappa)*min([1/(N+1)^2 (1-sqrt(kappa))^(2*(N+1))])
[double(psi) L/((1-kappa)*Ak(N+1)) ...
    L/(1-kappa)*min([1/(N+1)^2 (1-sqrt(kappa))^(2*(N+1))])]
    








