function [TSMI,maps,mu] = build_brainweb(Phi0, scale, L)
% A function to simulate the Time-Series of Magnetisation Images (TSMI)
% from ground-turth mixture maps given using the brainweb numerical phantom, and some assigned values for the t1/t2. 
%
% (c) 2020 Mohammad Golbabaee, Clarice Poon (University of Bath)
%%
% load the ground truth mixture maps
load('./data/brainweb_maps.mat');
[~,N,M] = size(maps);
maps = reshape(maps,3,[]); 
msk = maps > 0;

% Assigned tissues T1 T2 values
mu = [784, 77; ... % white matter
    1216, 96; ...  % grey matter
    4083, 1394];   % CSF

% build TSMI mixtures
scnn = @(x) bsxfun(@times,x,1./(scale));
TSMI = zeros(L, N*M);

for i=1:3   
    ind = find(msk(i,:)>0);
    s = numel(ind);
    g = repmat(mu(i,:),[s,1]);
    D = Phi0(scnn(g),1);
    D = D.M.';
    TSMI(:,ind) = bsxfun(@times, D, maps(i,ind)) + TSMI(:,ind);
end
TSMI = reshape(TSMI.',N,M,L);
