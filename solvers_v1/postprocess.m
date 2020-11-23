function [X,a] = postprocess(X,a , MC, Phi0, scale, TSMI, TSMI_norm, opt)
% This function chooses top compartments outputted by SGB-lasso for visualisation.
% MC: the number of compartments to visualise.
% opt: two options can be used for the selection (hard threhsolding) criteria: 
%   opt = 1  Selects MC most dominant compartments having the highest L2 norms for their mixture maps. (default option)
%   opt = 2  Average and assign the longest T1/T2 values to the CSF compartment, and
%            select the remainig MC-1 compartments having the highest L2
%            norms for their mimxture maps. (this option is not default, but can be sometimes useful for in-vivo data).
%
% After selecting these compartments (e.g. the matrix X containing tissues T1/T2 values), this program solves non-negative
% least squares (NNLS) to further fine-tune the corresponding mixture maps (the matrix a). 
%
% (c) 2020 Mohammad Golbabaee, Clarice Poon (University of Bath)
%%
MC = min(size(X,1),MC);  % Top MC compartments to visualize
n = size(TSMI,2);
scnn = @(x) bsxfun(@times,x,1./(scale));
options = optimset('TolX',1e-6,'Tolfun',1e-6,'Display','off'); % set accuracy of NNLS /in-built Matlab solver

%% Threhsold very small compartments 
thrsh = 0.01;

% normalise mixture weights (a) such that they sum to 1 per-pixel
D = Phi0(scnn(X),1);
D=(D.M).';
normD = sqrt(sum(abs(D).^2,1));
a_aux = a.* (1./normD(:));
a_aux = bsxfun(@times,a_aux,1./(sum(a_aux,1)));
a_aux(isnan(a_aux))=0;

Wsc = sqrt(sum(abs(a_aux).^2,2));
Wsc=Wsc/max(Wsc);

ind = find(Wsc>thrsh);
Xsc = X(ind,:);
Wsc = Wsc(ind,:);

X=Xsc;
a = a(ind,:);

MC = min(size(X,1),MC);  % Top MC compartments to visualize
%%
switch opt        
    case 1
        %--choose top energy spikes
        D = Phi0(scnn(X),1);
        D=(D.M).';
        normD = sqrt(sum(abs(D).^2,1));
        a_aux = a.* (1./normD(:));
        
        a_aux = bsxfun(@times,a_aux,1./sum(a_aux,1));
        a_aux(isnan(a_aux))=0;
        
        Energy = sqrt(sum(abs(a_aux).^2,2));
        [Energy,ind] = sort(Energy,'descend');
        
        X = X(ind(1:MC),:);
        a = a(ind(1:MC),:);
        
        %--Update thresholded weights via NNLS
        D = Phi0(scnn(X),1);
        D=(D.M).';
        normD = sqrt(sum(abs(D).^2,1));
        
        D1 = normc(D);
        D1(isnan((D1)))=0;
        
        a = zeros(MC,n);
        parfor p  =1:n
            a(:,p) = lsqnonneg(D1, TSMI(:,p), options);
        end
        
    case 2       
        %--choose top energy spikes
        D = Phi0(scnn(X),1);
        D=(D.M).';
        normD = sqrt(sum(abs(D).^2,1));

        a_aux= bsxfun(@times, a, 1./normD(:));
        
        a_aux = bsxfun(@times,a_aux,1./(sum(a_aux,1)));
        a_aux(isnan(a_aux))=0;
        
        Energy = sqrt(sum(abs(a_aux).^2,2));
        [Energy,ind] = sort(Energy,'descend');
        
        X1 = X(ind(1:MC-1),:);
        a1 = a(ind(1:MC-1),:);
        
        IND = find( (X(:,1)>max(X1(:,1))) .* (X(:,2)>max(X1(:,2))));  % Find longer compartments
        
        W = sqrt(sum(a(IND,:).^2,2)); W = W/sum(W);
        X0 = sum(X(IND,:).* repmat(W,1,2) ,1); % compute weighted mean of long compartmens for the CSF tissue.
        
        X = [X1;X0];
        %--Update thresholded weights via NNLS
        D = Phi0(scnn(X),1);
        D=(D.M).';
        normD = sqrt(sum(abs(D).^2,1));
        
        D1 = normc(D);
        D1(isnan((D1)))=0;
        
        a = zeros(MC,n);
        parfor p  =1:n
            a(:,p) = lsqnonneg(D1, TSMI(:,p), options);
        end
end
%% Re-scalings to get the final mixture factors/maps
YY = bsxfun(@times, TSMI, TSMI_norm);
% re-normalise back mixture weights (a) according to dictionary and TSMI norms
a_aux= bsxfun(@times, a, TSMI_norm);
a_aux= bsxfun(@times, a_aux, 1./normD(:));
a_aux(isnan(a_aux))=0;
a = a_aux;
a = bsxfun(@times,a, 1./(sum(a,1)) ); %sum to 1 per pixel
a(isnan(a))=0;