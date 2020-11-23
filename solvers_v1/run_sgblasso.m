function [X,a]= run_sgblasso(y,param)
% This program solves the sparse group lasso (SGB-Lasso) regularisation problem 
% using Frank-Wolfe iteration with application to the Multi-compartment MR Fingerprinting (MRF). 
% The SGBLasso optimisation problem reads:
% 
%       argmin_{X,a}  || y - Phi0(X)*a ||^2_F  +  alpha*(1-beta) ||a||_1   +  alpha*beta*sqrt(v) ||a||_1,2
%
%   inputs: - y: matrix of multichannel signals e.g. time-series of vectorised magnetisation
%                images (TSMI) in the MRF application.
%           - Phi0: the non-linear dictionary parametrised
%                   by X. In MRF application, Phi0 is the Bloch response
%                   function approximated by a neural nettwork. This response is paramtrised by 
%                   the NMR properties namely X=(T1,T2) relxation times.
%           - alpha: The overal regularisation parameter of the sparse group total variation norm.
%           - beta: internal parameter of the sparse group total variation norm. 
%                   beta is between 0 and 1. Small beta promotes sparsity within the mixture maps, 
%                   large beta estimates fewer/sparser groups of compartments. 
%                   (v is the number of channels in y e.g. number of pixels in TSMI).
%
%   outputs: - X: a matrix of NMR parameters in MRF application. Each row of X corresponds to an 
%                 estimated tissue compartment parametrised by T1,T2 relaxatioin times.  
%            - a : the matrix of mixture maps.
%
% This program uses the LBFGS optimisation toolbox which require to be
% installed separately and added to the Matlab path while running this
% function. An implementation of LBFGS can be found in: https://github.com/stephenbeckr/L-BFGS-B-C
%
% For greater details about the SGB-lasso regularisation and its application in multi-compartment MR Fingerprinting, 
% please see the following papers:
%       M. Golbabaee and C. Poon, "An off-the-grid approach to multi-compartment magnetic resonance fingerprinting", 2020.
%       C. Poon and M. Golbabaee, "The sparse-group Beurling-lasso", 2020.
%
% (c) Mohammad Golbabaee, Clarice Poon -- University of Bath, 2020
%% normalise TSMI
y = normc(y);
y(isnan(y))=0;

%% load dictionary
load(param.filepath.network)
Phi0 = @(x,mode) gen_mag_der_DL(x,nn,mode);
param.nnscale= double(nn.scale*1000);

%% set internal params
param.do_noncvx = 1;
param.localascent = 1;
param.enforce_constr = 1;
lambda = param.alpha;
do_noncvx = 1;
maxit = param.maxit;
eta_tol = param.eta_tol;
ista_maxit = 2000;
grid_res = param.grid_res;
L = param.L;
u = 1- param.beta; % regularisation weight
param.w1 = u; % the l1 weight 
param.w12 = (1-u)*sqrt(size(y,2));
if u==0
    param.Reg = 'grouplasso+';
elseif u==1
    param.Reg ='lasso+';
else
    param.Reg ='sparsegrouplasso+';
end
%%
y = y(:);
%%
if param.enforce_constr ==1
    bfgs_solver = 'lbfgs_constr';
else
    bfgs_solver = 'lbfgs';
end

%% construct useful operators
Ra = @(a,s) reshape(a,s,[]); %amplitude shapes, input to Phi_T, where |T| = s
Rx = @(X) reshape(X,[],2); %Theta shape
Ry = @(Y) reshape(Y,L,[]); %input to Phi^*
vec = @(x) x(:);
d = 2; % size of theta: 2 for parameters T1/T2
len = @(x) length(x(:))/d;

gM = @(x,mode) getPhiMtx_AD(x,Phi0,L,mode);
G = @(X,i) X{i};
Phi = @(a,T)  (G(gM(Rx(T),1),1))*Ra(a,len(T)); 
Phi_t = @(a,T) vec( ( G(gM(Rx(T),1),1) )'* Ry(a) );
Ft = @(x) x;
FtF = @(x) x;
C = @(U,X,a) Phi_t(FtF(Phi(a,X)),U);
Cxx0 = @(a,M) M'*FtF(M*a);
Cxx = @(X,a)  vec(Cxx0(Ra(a,len(X)),  G(gM(Rx(X),1),1) ));

%% inputs for LBFGS
switch param.Reg
    case 'lasso+'
        a_norm = @(a,s) sum(a(:));
        m_sign = @(a,T) ones(size(a(:)));
        m_max = 1;
    case 'grouplasso+'
        a_norm = @(a,s) sum( sqrt(sum(Ra(a,s).^2,2))  );
        m_sign = @(a,T) glasso_sign(a,T);
        m_max = 1;
    case 'sparsegrouplasso+'
        w1 = param.w1;
        w12 = param.w12;
        if w12 == 0
            error('w12 == 0, set Reg to lasso+')
        end
        a_norm = @(a,s) w12*sum( sqrt(sum(Ra(a,s).^2,2))  ) + w1*sum(a(:));
        m_sign = @(a,T) w12*glasso_sign(a,T) + w1*ones(size(a(:)));
        m_max = w12;
end

if param.enforce_constr
    %we optimise over (z,x2), where z = x1 - x2, here RP(z,x2) = (x1,x2)
    RP0 = @(x) [x(:,1)+x(:,2), x(:,2)];
    RP = @(x) RP0(Rx(x));
else
    RP = @(x) Rx(x);
end

Fty = Ft(y);

%|F(Phi(a,x)) - y|^2
fun_discrep = @(a,x) - 2* sum(vec(Phi(a,x)).*vec(Fty)) ...
    + norm(y(:))^2 + sum(vec(FtF(Phi(a,x))).*vec(Phi(a,x)));

%LBFGS operators
fnE = @(x,a,lambda) lambda* a_norm(a,len(x)) +  fun_discrep(a,x)/2;


nablaEa = @(x,a,lambda) ( Cxx(x,a)- Phi_t(Ry(Fty),x) ...
    +lambda* m_sign(a,len(x)) );

if param.enforce_constr
    Ex = @(a,M) [vec(sum(a.*((M{2})'* (Ry(FtF(M{1}*a)) - Ry(Fty)) ),2) ); ...
        vec(sum( a.*( (M{3}+M{2})'* (Ry(FtF((M{1})*a)) - Ry(Fty))),2))] ;
else
    Ex = @(a,M) [vec(sum( a.*( (M{2})'* (Ry(FtF((M{1})*a)) - Ry(Fty))),2)) ; ...
        vec(sum(a.*((M{3})'* (Ry(FtF(M{1}*a)) - Ry(Fty)) ),2) )] ;
end


op.E = @(x,a,x0,a0,lambda) fnE(RP(x),a,lambda);
op.nablaEa = @(x,a,x0,a0,lambda) nablaEa(RP(x),a,lambda);
op.nablaEx = @(x,a,x0,a0,lambda)  Ex(Ra(a,len(x)),gM(RP(x),2));

%% perform Frank-Wolfe iterations
a = [];
X = [];

%---logarithmic-scaled gridding (Fixed grid)-----------------
aG=10; bG=4000; % Range of T1 (sec)
u0 =ceil(aG*2.^(linspace(0,log2(bG/aG),grid_res))); 

aG=4; bG=2000; % Range of T2 (sec)
u1 =ceil(aG*2.^(linspace(0,log2(bG/aG),grid_res)));

[U1,U2] = meshgrid(u0,u1); % The search grid
iii = find(U1>U2); % Enforce T1>T2
u = [U1(iii)/param.nnscale(1),U2(iii)/param.nnscale(2)]; % The T1>T2 search grid


eta_max=0;
for k=1:maxit
    
    if ~isempty(a)
        zk = ( Ry(Fty) - Ry( FtF(Phi(a,X)) ) )/lambda;
    else
        zk =  Ry(Fty)/lambda;
    end
    etak = @(uu) Ra( Phi_t(zk,uu) , len(uu) );
        
    %% Add a new compartment X = (T1,T2) 
    % For this, first search for largest abs value on the grid 
    % and then fine-tune the maximiser by solving a non-linear optimisation (i.e. a local ascent step).
    switch param.Reg
        case 'lasso+'
            para.proxG = @(c,T) max(c-T,0);%wthresh(c,'s',T);
            para.objG = @(c) sum((c(:)));           
            eta_abs = etak(u);
            eta_prev = eta_max;
           
            [eta_max,max_idx] = max(eta_abs(:));
            [pos, posX]=ind2sub(size(eta_abs),max_idx);
            S = @(v) v(posX);
            fk = @(x) S(etak(x));
            zk = zk(:,posX);
        case 'grouplasso+'
            para.proxG = @(c,T) group_wthresh_pos(c,T);
            para.objG = @(c) sum(sqrt(sum((c).^2,2)));
            fk = @(x) max( etak(x) ,0);
            eta_abs = sqrt(sum(fk(u).^2, 2)); %MG added %add threshold due to pos. constr.
            eta_prev = eta_max;
            [eta_max,pos] = max(eta_abs);
            
        case 'sparsegrouplasso+'
            para.proxG = @(c,T) group_wthresh_pos(max(c-w1*T,0),w12*T);
            para.objG = @(c) w12*sum(sqrt(sum((c).^2,2))) +  w1*sum((c(:)));
            
            fk = @(x) max( etak(x) - w1,0);
            eta_abs = sqrt(sum(fk(u).^2, 2));
            eta_prev = eta_max;
            [eta_max,pos] = max(eta_abs);
    end
    
    xnew = u(pos,:);
    anew = etak(xnew);
    
    if param.localascent
        fprintf('running...local ascent\n')
        max_old = eta_max;  xold = xnew;
        switch param.Reg
            case {'sparsegrouplasso+','grouplasso+'}
                %minimise -|| fk(x) ||^2/2
                xnew = locMax_glasso(xnew,gM,fk,zk,param.enforce_constr, Rx, Ra, vec);
                eta_max = norm(fk(xnew), 'fro');
            case 'lasso+'
                %minimise -fk(x) where fk(x) = [etak(x)]_i, i= max_ind
                xnew = locMax_lasso(xnew,gM,fk,zk,param.enforce_constr, Rx);
                eta_max = fk(xnew);
            otherwise
                warning('local ascent step not implemented')                
        end        
        if max_old > eta_max
            warning('error in local ascent step')
            xnew = xold; eta_max = max_old;
        end
    end
    
    if (eta_max)/m_max<1+eta_tol || abs(eta_max-eta_prev)/eta_max<1e-8
        break
    end
    
    a = [a; 0*vec(anew).'];
    X = [X; vec(xnew).'];
    clear etak eta_abs zk fk
        
    %% Perform restarted-FISTA to update the mixture weights (matrix a)
    s = len(X);
    para.a_init = a;
    para.lambda = lambda;
    para.ista_maxit = ista_maxit;
    M=gM(Rx(X),1);
    M= M{1};
    para.gamma = 10/norm(M'*M);
    para.tol = 1e-8;
    para.backtrack = 1;
    para.positivity = 1;
    para.verbose = 0;
    
    para.gradF = @(a) - (M'*Ry(Fty) -M'*Ry(FtF(M*a)));
    para.objF = @(a) 1/2*norm(y(:))^2 - sum(Fty(:).*vec(M*Ra(a,s))) + sum(vec(M*Ra(a,s)).*vec(FtF(M*Ra(a,s))))/2;   
    
    fprintf('running...FISTA\n')
    a = run_FISTA_pos(para);

    %% Now Simulaneously optimise the mixture weights (matrix a) and the T1/T2 values (matrix X)
    fprintf('running...LBFGS\n')
    if do_noncvx
        options.bfgs_solver = bfgs_solver;
        if param.enforce_constr
            Xtmp =[X(:,1)-X(:,2),X(:,2)];
        else
            Xtmp = X;
        end
        [X,a,~] = noncvx_sparse_spikes_pos(op,lambda, [],[], Xtmp(:),a(:), options);
        X = RP(X);
        %         X = Rx(X);
        a = Ra(a,len(X));
    end

    res = fun_discrep(a,X);
    fprintf('F-W iter = %i, eta_max/m_max = %e,|y-Phix|^2 = %e\n',k,eta_max/m_max,sqrt(res)/norm(y));
     
end
if ~isempty(a)
    res = fun_discrep(a,X);
else
    res=0;
end
fprintf('F-W Terminted:: eta_max/m_max=%e,|y-Phix|^2=%e\n',(eta_max)/m_max,sqrt(res)/norm(y(:)));

end


%% ====useful signs
function sgn =  glasso_sign(a,s)
a = reshape(a,s,[]);
sgn = a./ repmat(sqrt(sum(a.^2,2)),[1,size(a,2)]);
sgn = sgn(:);
sgn(isnan(sgn)) = 0;
end

%% ========== useful proxes
function x = group_wthresh_pos(x,T)
x(x<0)=0;
no = sqrt(sum(x.^2,2));
x =  bsxfun(@times,x, max(no-T,0)./no);
x(isnan(x)) = 0;
end

%% functions for the local ascent step for eta_max
function xnew = locMax_glasso(x0,gM,etak,zk,enforce_constr, Rx, Ra, vec)
if enforce_constr
    Xtmp =[x0(1)-x0(2),x0(2)];
    RP0 = @(x) [x(:,1)+x(:,2), x(:,2)];
    RP = @(x) RP0(Rx(x));
    nabla = @(a,M) -[vec(sum( a.*( (M{2})'* zk),2)) ; ...
        vec(sum(a.*((M{3}+M{2})'* zk ),2) )] ;
else
    Xtmp = x0;
    RP = @(x) Rx(x);
    nabla = @(a,M) -[vec(sum( a.*( (M{2})'* zk),2)) ; ...
        vec(sum(a.*((M{3})'* zk ),2) )] ;
end

etaVal = @(x) - norm(etak(RP(x)),'fro')^2/2;
etaNabla = @(x)  nabla(Ra(etak(RP(x)),1),gM(RP(x),2));

fun = @(x)fminunc_wrapper( x, etaVal, etaNabla);
LB = zeros(2,1); % lower bound (t1>=t2>=0)
UB = inf(2,1); % there is no upper bound
opts    = struct( 'x0',Xtmp(:),'printEvery',100,'maxIts',5000);
[x, ~, ~] = lbfgsb(fun, LB, UB, opts );

if enforce_constr
    xnew = [x(1)+x(2), x(2)];
else
    xnew = x(:)';
end
end


function xnew = locMax_lasso(x0,gM,etak,zk,enforce_constr, Rx)

if enforce_constr
    Xtmp =[x0(1)-x0(2),x0(2)];
    RP0 = @(x) [x(:,1)+x(:,2), x(:,2)];
    RP = @(x) RP0(Rx(x));
    nabla = @(M) -[(M{2})'* zk ; (M{3}+M{2})'* zk ] ;
else
    Xtmp = x0;
    RP = @(x) Rx(x);
    nabla = @(M) -[(M{2})'* zk ; (M{3})'* zk ] ;
end

etaVal = @(x) - etak(RP(x));
etaNabla = @(x)  nabla(gM(RP(x),2));

fun = @(x)fminunc_wrapper( x, etaVal, etaNabla);
LB = zeros(2,1); % lower bound (t1>=t2>=0)
UB = inf(2,1); % there is no upper bound
opts    = struct( 'x0',Xtmp(:),'printEvery',100,'maxIts',5000);
[x, ~, ~] = lbfgsb(fun, LB, UB, opts );

if enforce_constr
    xnew = [x(1)+x(2), x(2)];
else
    xnew = x(:)';
end
end