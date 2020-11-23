function a = run_FISTA_pos(para)
% Restarted Fast Iterative Shrinkage Algorithm (FISTA):
%
% B. O’donoghue and E. Candes, “Adaptive restart for accelerated gradient
% schemes,” Foundations of computational mathematics, vol. 15, no. 3, pp.
% 715–732, 2015.
% 
% A. Beck and M. Teboulle, “A fast iterative shrinkage-thresholding algorithm
% for linear inverse problems,” SIAM journal on imaging sciences,
% vol. 2, no. 1, pp. 183–202, 2009.
%
% Restarted FISTA used here for oprimising the mixture weights subject to
% non-negativity constraint.
%
% (c) 2020 Mohammad Golbabaee, Clarice Poon (University of Bath)
%%
ista_maxit = para.ista_maxit;
gradF = para.gradF;
objF = para.objF;
objG = para.objG;
proxG = para.proxG;
lambda = para.lambda;
tol = para.tol;
backtrack = para.backtrack;

ai = para.a_init;
bi = ai; t = 1;

Ei = objF(ai)+ lambda* objG(ai);

if(~isfield(para,'gamma'))
    backtrack = 1;
    gamma = 1;
else
    gamma = para.gamma;
end

for it=1:ista_maxit
    aim = ai;
    bim = bi;
    tm = t;
    
    if backtrack
        %backtrack for gamma
        beta = 0.5; %gamma = 1;
        grad = gradF(bi);
        
        ai = proxG(bi - gamma*grad, gamma*lambda);        
        objj=objF(bi);        
        while objF(ai) > objj + 1/(2*gamma)*norm(ai(:)-bi(:))^2 + real(grad(:)'*(ai(:)-bi(:)))%   delta*gamma*sum(gradF(bi).^2)
            %fprintf('reduce fista step...\n');
            gamma = beta*gamma;
            ai = proxG(bi - gamma*grad, gamma*lambda);            
        end        
    else
        ai = bi - gamma*gradF(bi);
        ai = proxG(ai,gamma*lambda);
    end

    t = (1+sqrt(1+4*tm^2))/2;
    v = (tm-1)/t;
    bi = ai+v*(ai - aim);

    if sum((bim(:) - ai(:)).*(ai(:) - aim(:)))>0
        bi = ai; t=1;
        %fprintf('restart\n');
    end    
    
    Enew = objF(ai)+ lambda* objG(ai);    
    if abs(Ei-Enew)/Ei <tol
        ai = aim;
        break;
    end
    if para.verbose ==1
        fprintf('iter=%i, fistaobj=%e\n',it, Enew);
    end
    Ei = Enew;
end
a = ai;
end

