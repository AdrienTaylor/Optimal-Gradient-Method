clc; clear all;
% In this code, we reproduce exactly the design procedure presented in
% Section 3.3 of the paper.

%% Part 1: setting up parameters and notations

% PARAMETERS TO BE TUNED 
verbose = 0; % let solver talk?
N       = 2;
L       = 1;
m       = .1;
cw = 1;
cf = 0;

% END OF TUNABLE ZONE
kappa = m/L;
w0       = zeros(N+1,1); w0(1,1)  = 1;
gk       = zeros(N+1,N); % each column is a g_i: gk = [g_0 g_1 ... g_{N-1}]
gk(2:end,:) = eye(N);
fk       = eye(N);       % each column is a f_i: fk = [f_0 f_1 ... f_{N-1}]

g        = @(i)(gk(:,i+1));% short cut for starting i-indexing of g at 0
f        = @(i)(fk(:,i+1));% short cut for starting i-indexing of f at 0

symmetrize = @(X)(1/2*(X+X')); % this is for symmetrizing the matrix S''
% (it is directly symmetrized in the paper,
% we show here a more direct way).

%% Part 2: write & solve the SDP
if N > 1 % we follow the same steps as in the paper, here (case N == 1 below)
    alpha_prime = sdpvar(N,N,'full');
    ap          = @(i,j)(alpha_prime(i,j+1)); % short cut for starting j-indexing of ap at 0
    
    lambdai_ip1 = sdpvar(1,N-1);% this is \lambda_{i,i+1} (i=0,...,N-2)
    lambdas_i   = sdpvar(1,N);  % this is \lambda_{*,i} (i=0,...,N-1)
    lambdaN_s   = sdpvar(1);    % this is \lambda_{N-1,*}
    
    % for notational convenience, we shift the indices
    lami_ip1    = @(i)(lambdai_ip1(i+1));
    lams_i      = @(i)(lambdas_i(i+1));
    lamN_s      = lambdaN_s;
    
    tau         = sdpvar(1);
    
    wN          = w0;
    for i = 0:N-1
        wN = wN - (ap(N,i)/L * g(i)+kappa*ap(N,i)*w0);
    end
    
    % define S'': (we do not symmetrize it here, but only later, for
    % simplicity)
    Spp = tau * (w0*w0')+ lamN_s/2/(L-m)*g(N-1)*g(N-1)';
    for i = 0:N-1
        Spp = Spp + lams_i(i)/2/(L-m)*g(i)*g(i)';
    end
    for i = 0:N-2
        Spp = Spp + lami_ip1(i)/2/(L-m)*(g(i)-g(i+1))*(g(i)-g(i+1))';
    end
    Spp = Spp - lams_i(0) * g(0)*w0';
    
    for i = 1:N-2
        coef = lami_ip1(i);
        for j = 0:i-1
            coef = coef - kappa * ap(i,j);
        end
        Spp = Spp - coef * g(i)*w0';
    end
    
    for i = 1:N-2
        for j = 0:i-1
            Spp = Spp + ap(i,j)/L * (g(i)*g(j)');
        end
    end
    
    coef = lamN_s;
    for j = 0:N-2
        coef = coef - kappa * ap(N-1,j);
    end
    Spp = Spp - coef * g(N-1)*w0';
    
    for j = 0:N-2
        Spp = Spp + ap(N-1,j)/L * g(N-1)*g(j)';
    end
    
    for i = 0:N-2
        coef = lami_ip1(i);
        for j = 0:i-1
            coef = coef - kappa * ap(i,j);
        end
        Spp = Spp + coef * g(i+1)*w0';
    end
    for i = 0:N-2
        for j = 0:i-1
            Spp = Spp - ap(i,j)/L * g(i+1)*g(j)';
        end
    end
    
    S = symmetrize(Spp);
    bigS = [S wN; wN' 1];
    cons = (bigS >= 0);
    
    linEq = - lamN_s * f(N-1);
    for i = 0:N-1
        linEq = linEq + lams_i(i) * f(i);
    end
    for i = 0:N-2
        linEq = linEq + lami_ip1(i) * (f(i+1)-f(i));
    end
    cons = cons + ( lambdai_ip1 >= 0);
    cons = cons + ( lambdas_i   >= 0);
    cons = cons + ( lambdaN_s   >= 0);
    cons = cons + ( linEq == 0 );
else % if N == 1, the SDP is simpler
    tau     = sdpvar(1);
    lamS0   = sdpvar(1); % this is lambda_{*,0}
    lam0S   = sdpvar(1); % this is lambda_{0,*}
    alpha_prime = sdpvar(1);
    
    w1      = w0 * (1-kappa*alpha_prime) - alpha_prime * g(0);
    
    LMI_cons = tau * w0*w0'  + lam0S/2/(L-m) * g(N-1)*g(N-1)'+lamS0/2 * g(0)*g(0)'/(L-m)-lamS0/2 * (w0*g(0)'+g(0)*w0');
    LIN_cons = -lam0S * f(0) + lamS0 * f(0);
    
    S = [LMI_cons w1; w1' 1];
    
    cons = (lamS0>=0);
    cons = cons + (lam0S>=0);
    cons = cons + (LIN_cons == 0);
    cons = cons + (S >= 0);
    
end

solver_opt      = sdpsettings('solver','mosek','verbose',verbose);
solverDetails   = optimize(cons,tau,solver_opt);

%% Part 3: recover alpha's and h's.

alpha = zeros(N,N);

for i = 1:N
    for j = 1:i
        if i == N
            alpha(i,j) = double(alpha_prime(i,j));
        elseif i==N-1
            alpha(i,j) = double(alpha_prime(i,j)/lamN_s);
        else
            alpha(i,j) = double(alpha_prime(i,j)/lami_ip1(i));
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
fprintf(' ||w_N-w_*||^2 / || w_0-w_*||^2 <= %6.5f \n',double(tau))
fprintf('Step sizes (using notations w_k = w_{k-1} - sum_i h_{k,i} f''(w_i))\n');
h
fprintf('Aggregated step sizes (using notations w_k = w_0 - sum_i h_{k,i} f''(w_i))\n');
h_aggr
fprintf('Step sizes (using alpha''s notations, see paper)\n');
alpha
fprintf('******************************************************************\n');
















