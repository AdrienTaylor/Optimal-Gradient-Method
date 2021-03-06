clc; clear all;
% In this code, we reproduce exactly the design procedure presented in
% Appendix D of the paper.

%% Part 1: setting up parameters and notations

% PARAMETERS TO BE TUNED 
verbose = 0; % let solver talk?
N       = 5;
L       = 1;
m       = 0.1;
% (A) for reproducing results from Appendix E.1 for (f(w_N)-f_*)/||w_0-w_*||^2, set to cw=1, cf=0
% (B) for reproducing results from Appendix E.2 for (f(w_N)-f_*)/(f(w_0)-f_*),  set to cw=0, cf=1
cw      = 1;    
cf      = 0;

% END OF TUNABLE ZONE
kappa = m/L;
w0       = zeros(N+2,1); w0(1,1)  = 1;
gk       = zeros(N+2,N+1); % each column is a g_i: gk = [g_0 g_1 ... g_{N}]
gk(2:end,:) = eye(N+1);
fk       = eye(N+1);       % each column is a f_i: fk = [f_0 f_1 ... f_{N}]

g        = @(i)(gk(:,i+1));% short cut for starting i-indexing of g at 0
f        = @(i)(fk(:,i+1));% short cut for starting i-indexing of f at 0

symmetrize = @(X)(1/2*(X+X')); % this is for symmetrizing the matrix S''
% (it is directly symmetrized in the paper,
% we show here a more direct way).

%% Part 2: write & solve the SDP
beta = sdpvar(N,N,'full');
bp          = @(i,j)(beta(i,j+1)); % short cut for starting j-indexing of bp at 0

lambdai_ip1 = sdpvar(1,N);    % this is \lambda_{i,i+1} (i=0,...,N-1)
lambdas_i   = sdpvar(1,N+1);  % this is \lambda_{*,i} (i=0,...,N)

% for notational convenience, we shift the indices
lami_ip1    = @(i)(lambdai_ip1(i+1));
lams_i      = @(i)(lambdas_i(i+1));
tau         = sdpvar(1);

wN          = w0;
for i = 0:N-1
    wN = wN - (bp(N,i)/L * g(i)+kappa*bp(N,i)*w0);
end

% define \bar{S}'': (we do not symmetrize it here, but only later, for
% simplicity)
Sbar_pp = tau * (cw+m/2*cf)*(w0*w0');
for i = 0:N
    Sbar_pp = Sbar_pp + lams_i(i)/2/(L-m)*g(i)*g(i)';
end
for i = 0:N-1
    Sbar_pp = Sbar_pp + lami_ip1(i)/2/(L-m)*(g(i)-g(i+1))*(g(i)-g(i+1))';
end
Sbar_pp = Sbar_pp - lams_i(0) * g(0)*w0';

for i = 1:N-1
    coef = lami_ip1(i);
    for j = 0:i-1
        coef = coef - kappa * bp(i,j);
    end
    Sbar_pp = Sbar_pp - coef * g(i)*w0';
end

for i = 1:N-1
    for j = 0:i-1
        Sbar_pp = Sbar_pp + bp(i,j)/L * (g(i)*g(j)');
    end
end

coef = 1;
for j = 0:N-1
    coef = coef - kappa * bp(N,j);
end
Sbar_pp = Sbar_pp - coef * g(N)*w0';

for j=0:N-1
    Sbar_pp = Sbar_pp + bp(N,j) * g(N)*g(j)';
end

for i = 0:N-1
    coef = lami_ip1(i);
    for j = 0:i-1
        coef = coef - kappa * bp(i,j);
    end
    Sbar_pp = Sbar_pp + coef * g(i+1)*w0';
end
for i = 0:N-1
    for j = 0:i-1
        Sbar_pp = Sbar_pp - bp(i,j)/L * g(i+1)*g(j)';
    end
end

S = symmetrize(Sbar_pp);
bigS = [S sqrt(m)*wN; sqrt(m)*wN' 2];
cons = (bigS >= 0);

linEq = tau*cf * f(0) - f(N);
for i = 0:N-1
    linEq = linEq + lami_ip1(i) * (f(i+1)-f(i));
end
for i = 0:N
    linEq = linEq + lams_i(i) * f(i);
end
cons = cons + ( lambdai_ip1 >= 0);
cons = cons + ( lambdas_i   >= 0);
cons = cons + ( linEq == 0 );

solver_opt      = sdpsettings('solver','mosek','verbose',verbose);
solverDetails   = optimize(cons,tau,solver_opt);

%% Part 3: recover alpha's and h's.

alpha = zeros(N,N);

for i = 1:N
    for j = 1:i
        if i == N
            alpha(i,j) = double(beta(i,j));
        else
            alpha(i,j) = double(beta(i,j)/lami_ip1(i));
        end
    end
end

% compute h's

h = zeros(N,N);

for k = N:-1:1
    h(k,k) = alpha(k,k);
    for i = k-2:-1:0
        h(k,i+1) = alpha(k,i+1) - alpha(k-1,i+1);
        for j = i+1:k-1
            h(k,i+1) = h(k,i+1) + m/L * h(k,j+1)*alpha(j,i+1);
        end
    end
end


% compute aggregated h's
h_aggr = zeros(N,N);
h_aggr(1,:) = h(1,:);
for i = 2:N
    h_aggr(i,:) = h_aggr(i-1,:) + h(i,:);
end





%% Part 4: summary

fprintf('******************************************************************\n');
fprintf('Design procedure for N=%d: summary \n',N)
fprintf('Solver status: %s\n',solverDetails.info)
fprintf('Worst-case guarantee of the optimized step sizes:\n');
fprintf(' (f(w_N)-f_*) / (c_w || w_0-w_*||^2 + c_f (f(w_0)-f_*)) <= %6.5f [with cw=%3.2f, cf=%3.2f]\n',double(tau),cw,cf)
fprintf('Step sizes (using notations w_k = w_{k-1} - sum_i h_{k,i} f''(w_i))\n');
h
fprintf('Aggregated step sizes (using notations w_k = w_0 - sum_i h_aggr_{k,i} f''(w_i))\n');
h_aggr
fprintf('Step sizes (using alpha''s notations, see paper)\n');
alpha
fprintf('******************************************************************\n');
















