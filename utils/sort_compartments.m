function [E,ind] = sort_compartments(X,X_GT)
% Sort recovered T1/T2 compartments to have lowest Mean Avegrage Percentage Error (MAPE) to the ground-truth.
% (c) 2020 Mohammad Golbabaee, Clarice Poon (University of Bath)
%%
if size(X,1)~=size(X_GT,1)
    error('inputs dim mismatch')
end
p = perms([1:size(X,1)]);
% check all permulations for lowest error
F = inf;
for k=1:size(p,1)    
    e = abs(X(p(k,:),:)-X_GT)./X_GT;        
    if sum(e(:))<F
        F = sum(e(:));
        E = e;
        ind=p(k,:);
    end    
end
