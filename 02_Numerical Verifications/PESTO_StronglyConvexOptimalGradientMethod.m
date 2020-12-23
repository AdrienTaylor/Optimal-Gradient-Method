clear all; clc;
% In this example, we use the optimal gradient method for solving the 
% L-smooth mu-strongly convex minimization problem
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
N = 1;		% number of iterations

Akn = @(x) ( ( (1+kappa)*x+2*(1+sqrt((1+x)*(1+kappa*x)) ))/(1-kappa)^2); 
Akv = zeros(N+1,1);
for i = 1:N
    Akv(i+1) = Akn(Akv(i));
end
Ak  = @(k)(Akv(k+1));
bk  = @(k)(Ak(k)/(1-kappa)/Ak(k+1));
etk = @(k)(1/2 * ( (1-kappa)^2*Ak(k+1) -(1+kappa)*Ak(k))/(1+kappa+kappa*Ak(k)) );

% Algorithmic parameters
y = cell(N+1,1);% store iterates in a cell
x = cell(N+1,1);% store iterates in a cell
z = cell(N+1,1);% store iterates in a cell
g = cell(N+1,1);% store iterates in a cell
f = cell(N+1,1);% store iterates in a cell

y{1}    = x0;
x{1}    = x0;

z{1}    = x0;

for i=0:N-1
    y{i+2}  = bk(i) * x{i+1} + (1-bk(i)) * z{i+1};
    [g{i+2},f{i+2}] = F.oracle(y{i+2});
    x{i+2}  = y{i+2} - 1/L * g{i+2};
    z{i+2}  = z{i+1} + kappa*etk(i) * ( y{i+2} - z{i+1}) - etk(i)/L * g{i+2};
end

% (4) Set up the performance measure
obj = ( z{N+1} - xs)^2;
P.PerformanceMetric(obj); % Worst-case evaluated as F(x)-F(xs)

% (5) Solve the PEP
P.TraceHeuristic(1);
P.solve()

% (6) Evaluate the output
[double(obj) 1/(1+kappa*Ak(N))]
    








