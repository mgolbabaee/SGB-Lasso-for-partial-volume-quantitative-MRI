function [x,a,R] = noncvx_sparse_spikes_pos(op,lambda, x0,a0, x1,a1, options)
% [xf,af] = noncvx_sparse_spikes(op,x0,a0, x1,a1)
%   Solve
%       min_{x,a} 1/(2*lambda)|Phi_x0*a0-Phi_x*a|^2 + |a|_1
%   using L-BFGS with initialization (x1,a1).
%
%   Copyright (c) 2017 Gabriel Peyre
%%
options.null = 0;
bfgs_solver = options.bfgs_solver;

op.xlim = [0,0];
op.ylim = [Inf,Inf];
X = @(z)z(1:length(x1));
A = @(z)z(length(x1)+1:end);
XA = @(x,a)[x(:);a(:)];

z1 = XA(x1,a1);
Ebfgs = @(z)op.E(X(z),A(z),x0,a0,lambda);
nablaE = @(z)real( XA( op.nablaEx(X(z),A(z),x0,a0,lambda), op.nablaEa(X(z),A(z),x0,a0,lambda) ) );
% callback for L-BFGS
nablaEbfgs = @(z)deal(Ebfgs(z),nablaE(z));

switch bfgs_solver
    case 'lbfgs'
        % Becker's v3 lbfs
        fun     = @(x)fminunc_wrapper( x, Ebfgs, nablaE);
        Nx = length(x1);
        N = length(a1);
        l  = zeros(Nx+N,1) ;              % lower bound (positive t1/t2)
        u  = inf(Nx+N,1);                 % there is no upper bound
        opts    = struct( 'x0',z1, 'printEvery',100,'maxIts',1000, 'm', 100);
        [z1, R, info] = lbfgsb(fun, l, u, opts );
        
    case 'lbfgs_constr'
        Nx = length(x1);
        N = length(a1);
        LB = zeros(Nx+N,1) ;              % lower bound (t1>=t2>=0) and a>=0
        UB = inf(Nx+N,1);                 % there is no upper bound        
        % Becker's v3 lbfs
        fun     = @(x)fminunc_wrapper( x, Ebfgs, nablaE);
        opts    = struct( 'x0',z1,'printEvery',100,'maxIts',1000, 'm', 100);
        [z1, R, info] = lbfgsb(fun, LB, UB, opts );        
end
x = X(z1); a = A(z1);

end