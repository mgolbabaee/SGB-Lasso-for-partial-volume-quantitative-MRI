% Multicomparment (partial volume) quantitative MRI reconstruction demo
%
% Algorithm used: the Sparse Group Beuling Lasso (SGB-Lasso) implemented via Frank-Wolfe iterations in the following paper:
%       M. Golbabaee and C. Poon, "An off-the-grid approach to multi-compartment magnetic resonance fingerprinting", 2020.
% Data used: Brainweb numerical phantom and fuzzy segmented white matter, grey matter and CSF compartments.
%
% (c) 2020 Mohammad Golbabaee, Clarice Poon (University of Bath)
%%
clc;
clear;
close all;
addpath(genpath('./'));
%% Load a trained neural network that approximates the Bloch Responses in a magnetic resonance fingerprinting (MRF) application
% For training this network given a MRF dictionary, see the file below and train only a Decoder: 
% https://github.com/mgolbabaee/LRTV-MRFResnet-for-MRFingerprinting/blob/master/train_mrfresnet/Train_MRFResnet.m

filepath.network = './data/bloch/BlochNetwork_2_500_10.mat';
load(filepath.network);

%% Create the QMRI operator: a continous dictionary of Bloch responses and its derivatives. 
% in this demo the Bloch response is a non-linear function of two paramters (T1, T2 relaxation times) approximated by a neural network.
Phi0 = @(x,mode) gen_mag_der_DL(x,nn,mode); % This function depending on mode, generates either the nonlinear Bloch responses or their derivatives (Jacobian) for a given theta (set of T1/T2 values)

L = 10; % The length of the Bloch responses. In this demo the dimension-reduced MRF response has a dimenion 10.
scale = nn.scale*1000; % The T1/T2 values were down-scaled by these values when training the neural network (the network inputs).

%% load Time-Series of Magnetisation Images (TSMI)
% TSMIs should normmally be reocnstructed from k-space data first, e.g. for MRF applications one can use algorithms 
% that exploit subspace dimensionality-reduction such as https://github.com/mgolbabaee/LRTV-MRFResnet-for-MRFingerprinting.
%
% In this demo we skip the k-space reconstruction step and directly form a (dimension-reduced) TSMI phantom using the Brainweb's fuzzy segmented maps. 
% This phantom has three mixed compartments corresponding to the white matter, grey matter and the CSF tissues.

[TSMI, maps, theta] = build_brainweb(Phi0, scale, L); % Outputs the TSMI, the ground-truth mixture maps (maps), and their corresponding T1/T2 values (theta).
[N,M,L] = size(TSMI);

%% Visualise ground truth compartments
figure(1); clf
for k =1:size(maps,1)
    subplot(1,size(maps,1),k);
    imagesc(reshape(maps(k,:),N,M)); axis off
    title(sprintf('T1 true: %2.2f (ms)\nT2 true: %2.2f (ms)\n',theta(k,1),theta(k,2))); 
end

%% pre-processing
%---Mask the forground
thresh = 0;
mask =  sqrt(sum(abs(TSMI).^2,3))>thresh;
TSMI = reshape(TSMI,[],L);
TSMI = TSMI( mask(:),:);

%---phase correction step
ph = angle(TSMI(:,1));
TSMI = bsxfun(@times, TSMI, exp(-1j*ph));
TSMI = real(TSMI)';
TSMI = double(TSMI);

%--- normalisation step
TSMI_norm = sqrt(sum(abs(TSMI).^2,1));
TSMI = normc(TSMI);
TSMI(isnan(TSMI))=0;

%% Solve SGB-Lasso for compartment separation (main processing)
opts.filepath = filepath; 
opts.L = 10;            % Sequence legth
opts.grid_res = 10;     % Size of the grid-search step for descretising the T1/T2 space.
opts.maxit= 20;         % Maximum number of iterations (default 20). 
opts.eta_tol = 1e-1;    % Convergence tolerance parameter (default 1e-1).
opts.beta = 1e-3;       % 1-beta is the regularisaion weight corresponding to the L1 term (beta is a value between 0 to 1).  
opts.alpha = .8;        % the overall regularisation parameter

[theta_rec, maps_rec] = run_sgblasso(TSMI,opts); % run the SGB-Lasso algorithm for compartment separation
theta_rec = bsxfun(@times, theta_rec, scale);    % rescale the estimated T1/T2 values in msec

%% Hard threhshold a desired number of estimated compartments (post-processing) and visualise them
MC = 3;      % Number of compartments returned after a hard thresholding step.
option = 1;  % Select most dominant compartments (having the highest L2 norm for their mimxture maps).
[theta_rec, maps_rec] = postprocess(theta_rec, maps_rec, MC, Phi0, scale, TSMI, TSMI_norm, option); 

%-- Show estuimated T1/T2 values and mixture maps 
MC = min(size(theta_rec,1), MC); 

figure(2); clf
tmp = zeros(MC, N*M);
tmp(:,mask) = maps_rec; 
maps_rec = tmp.';
maps_rec = reshape(maps_rec,N,M,[]);

for k =1:MC
    subplot(1,MC,k);
    imagesc((maps_rec(:,:,k))); axis off
    title(sprintf('T1 est: %2.2f (ms)\nT2 est: %2.2f (ms)\n',theta_rec(k,1),theta_rec(k,2))); 
end