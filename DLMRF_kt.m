
clear;
%% Set some parameters for numerical simulation
R = 5;    % Rank of approximation
nx = 256;  % Image size
ny = nx;   % Image size
cm = 3;

snr = 30;  % SNR = 20log(s/sigma), s = 0.08716;  (optional: 20 25 30 35 40)
sigma = 0.06/(10^(snr/20));

% This addpath assumes that you are in the example folder. Otherwise make
% sure the whole toolbox is in your pathdef and forget about this line.
addpath(genpath('./utils'))
addpath(genpath('./fessler_nufft'))

%% Create Dictionary
load('flip_angle_pattern.mat')
TR0 = 4e-3;          % Basis TR used for the TR pattern
pSSFP = 1;           % Boolean (TR = TR0, if pSSFP == 0)

load('Dictionary_Tissue_Par.mat')
idx = 1:cm:850;
D = MRF_dictionary(T1, T2, [], alpha, TR0, pSSFP, idx, R);
D_index = [];
D_num = 0;
for i = 1:size(D.lookup_table,1)
    if D.lookup_table(i,2) < D.lookup_table(i,1)
        D_num = D_num+1;
        D_index(D_num) = i;
    end
end
D.magnetization = D.magnetization(:,D_index);
D.z = D.z(:,D_index);
D.normalization = D.normalization(:,D_index);
D.lookup_table = D.lookup_table(D_index,:);

% and add some optional parameters used for plotting in admm_recon.m
D.plot_details{1} = 'title(''T1 (s)''); caxis([0.2  5]); colormap hot';
D.plot_details{2} = 'title(''T2 (s)''); caxis([0 .5]); colormap hot';
D.plot_details{3} = 'title(''PD (a.u.)''); caxis([0.5 1]); colormap gray';

%% do numerical simulation to obtain undersampled k-space data
load('MRF_Brain.mat')
MRF = MRF(:,:,idx);
nt = size(MRF,3);    % Number of time frames

load('k_tj2.mat')
k(:,1,:) = real(k_tj)*pi/5;
k(:,2,:) = imag(k_tj)*pi/5;
k = repmat(k,[1 1 18]);
k = k(:,:,1:nt);

% dcf = repmat(dcf,[1 18]);
% dcf = dcf(:,1:nt);
% dcf = col(dcf);

% nonuniform fast Fourier transform operator
E = LR_nuFFT_operator(k, [nx ny], [], [], 2, [], 'minmax:kb'); % this nuFFT operator does not incorporate a low rank transformation
ELR = LR_nuFFT_operator(k, [nx ny], D.u, [], 2, [], 'minmax:kb'); % LR NuFFT operator

% Simulate noisey data:
Y = E*MRF;
Y = Y + sigma*(randn(size(Y))+1i*randn(size(Y)));

%% set parameters for coupled dictionary learning
numIter = 60;
ErrTh_start = 0.1; % error threshold (RMSE) in the beginning, % RMSE = 10^(-PSNR_goal/20);
ErrTh_end = 0.03;  % error threshold at the end	
errNum = 40; % number of outer iterations with error threshold decreasing linearly from ErrTh_start to ErrTh_end.
errthArray = linspace(ErrTh_start, ErrTh_end, errNum); % linearly decreasing error thresholds.
errthArray = [errthArray, ErrTh_end*ones(1, numIter-errNum)];

DLMRIparams.numIter = numIter; % number of outer iterations.
DLMRIparams.R = R;
DLMRIparams.n = 25; % length of an atom, i.e., the number of pixels in an image patch.
DLMRIparams.K = 512; % number of atoms in a coupled dictionary.
DLMRIparams.N = 10000; % number of samples for dictionary training.
DLMRIparams.s = 20; % sparsity constraint, round((0.15)*DLMRIparams.n)
DLMRIparams.numIter_KSVD = 50; % number of inner iterations for dictionary learning
DLMRIparams.r = 1; % distance between two adjacent image patches.
DLMRIparams.variance_thresh = -1; % Discard those patch pairs with too small variance during training.

DLMRIparams.lambda = 5e-3; % balance the data fidelity term and sparisity term 
DLMRIparams.cg_iter = 20; % number of iterations for conjuate gradient algorithm.

%% start coupled dictionary learning for MRF
RE1 = [];
RE2 = [];
RE3 = [];

%initialize X with gridding reconstruction result
X_iter = zeros(nx,ny,R); 

% CDL-MRF iterations
for niter = 1:DLMRIparams.numIter
     %% step 1: dictionary matching 
    X = X_iter;
    X = reshape(X, [nx*ny R]);
    clear c idx
    for q=size(X,1):-1:1 
        [c(q,1),idx(q,1)] = max(X(q,:) * conj(D.magnetization), [], 2);
    end
    PD = c ./ D.normalization(idx).';
    qMaps = D.lookup_table(idx,:);
    qMaps = reshape(qMaps, [nx, ny, size(D.lookup_table,2)]);
    qMaps(:,:,3) = reshape(PD, [nx, ny]);
    
    X = D.magnetization(:,idx).*(D.normalization(:,idx).*abs(PD).');  
    X = reshape(X.',[nx ny R]);
    
     %% step 2: coupled dictionary learning
    %L2 normalization for each contrast images in X
    X = reshape(X, [nx*ny R]);
    if(niter~=1)
        norm_X = max(abs(X),[],1);
        X = X./repmat(norm_X, [nx*ny 1]);
    end
    X = reshape(X,[nx ny R]);
    
    %creating image patches
    clear blocks blocks_idx
    for ns = 1:R
        [blocks{ns},blocks_idx{ns}] = my_im2col(X(:,:,ns),[sqrt(DLMRIparams.n),sqrt(DLMRIparams.n)],DLMRIparams.r); 
        mean_blocks{ns} = mean(blocks{ns});
        blocks_sub_mean{ns} = blocks{ns}-repmat(mean_blocks{ns},[DLMRIparams.n,1]); %subtract means of patches
        [rows{ns}, cols{ns}] = ind2sub([nx,ny]-sqrt(DLMRIparams.n)+1, blocks_idx{ns});
    end
    N_blocks = size(blocks{1},2); %total number of overlapping image patches
    Trainingdata_index = randperm(N_blocks);
    
    %select training data - using random selection/subset of patches
    for ns = 1:R
        Training_data{ns} = blocks_sub_mean{ns}(:,Trainingdata_index(1:DLMRIparams.N));
    end
     
    %discard those patch pairs with too small variance when construct training dataset
    clear variance_index;
    for ns = 1:R
        variance_index{ns}  = (sum(Training_data{ns}.^2, 1) > DLMRIparams.variance_thresh);
    end
    
    vi = ones(1,DLMRIparams.N);
    for ns = 1:R
       vi = vi & variance_index{ns};
    end
%     vi = zeros(1,DLMRIparams.N);
%     for ns = 1:R
%        vi = vi | variance_index{ns};
%     end   
    
    for ns = 1:R
        Training_data{ns} = Training_data{ns}(:,vi);
    end
     
    % dictionary initialization for dictionaries: random patches
    iniDictionary_coupled = {};
    initCoupled_index = randperm(size(Training_data{1},2));
    for ns = 1:R
        iniDictionary_coupled{ns} = Training_data{ns}(:,initCoupled_index(1:DLMRIparams.K));
    end
    
    % dictionary learning using modified K-SVD
    DLMRIparams.iniDictionary_coupled = iniDictionary_coupled;
    DLMRIparams.thr = errthArray(niter); % error thresholds for sparse coding during reconstruction.
    Dict = KSVD_DL_kt(Training_data, DLMRIparams);
    
     %% step 3: Computing sparse representations of all patches and summing up the patch approximations
    nc_atoms = 0;
    X_S = zeros(nx,ny,R);
    weight = zeros(nx,ny,1); 
    for jj = 1:10000:N_blocks
        jumpSize = min(jj+10000-1,N_blocks);
        SparseCoding_input = [];
        for ns = 1:R
            SparseCoding_input = [SparseCoding_input; blocks_sub_mean{ns}(:,jj:jumpSize)];
        end
        Coefs = OMPerrn(Dict,SparseCoding_input,DLMRIparams.thr,DLMRIparams.s);
        SparseCoding_output = Dict*Coefs;
        
        nc_atoms = nc_atoms+length(find(Coefs));
        
        blocks_tmp = {};
        for ns = 1:R
            blocks_tmp{ns} = SparseCoding_output(1+(ns-1)*DLMRIparams.n:ns*DLMRIparams.n,:) + repmat(mean_blocks{ns}(jj:jumpSize), [DLMRIparams.n 1]);
        end
        
        %summing up patch approximations
        for i  = jj:jumpSize
            col = cols{1}(i);
            row = rows{1}(i);
            for ns = 1:R
                X_S(row:row+sqrt(DLMRIparams.n)-1,col:col+sqrt(DLMRIparams.n)-1,ns) = X_S(row:row+sqrt(DLMRIparams.n)-1,col:col+sqrt(DLMRIparams.n)-1,ns) + reshape(blocks_tmp{ns}(:,i-jj+1),[sqrt(DLMRIparams.n) sqrt(DLMRIparams.n)]);
            end
            weight(row:row+sqrt(DLMRIparams.n)-1,col:col+sqrt(DLMRIparams.n)-1,:)=weight(row:row+sqrt(DLMRIparams.n)-1,col:col+sqrt(DLMRIparams.n)-1,:)+ones(sqrt(DLMRIparams.n));
        end
    end
        
    %patch-averaged result
    X_S = X_S./repmat(weight,[1 1 R]);
        
    %multiply magnitude
    if(niter~=1)
        norm_X = reshape(norm_X,[1 1 R]);
        X_S = X_S.*repmat(norm_X,[nx ny 1]);
    end
    
    % compute average used atoms
    nc_atoms = nc_atoms/N_blocks;
    disp([num2str(niter) ' iterations completed，平均原子数为：' num2str(nc_atoms)]);
    
     %% step 4: enforce k-space consistency with CG algorithm
    v = ELR'*Y + DLMRIparams.lambda*X_S;
    f = @(u) ELR'*(ELR*u) + DLMRIparams.lambda*u;
    X_iter = conjugate_gradient(f,v,1e-6,DLMRIparams.cg_iter,X_iter,0);
    
     %% caculate relative error and plot
    load('Brain.mat')
    mask = mask(:);
    T1gt = T1gt(:);
    T2gt = T2gt(:);
    PDgt = PDgt(:);
    
    T1_recon = qMaps(:,:,1);
    T2_recon = qMaps(:,:,2);
    PD_recon = qMaps(:,:,3);
    T1_recon = T1_recon(:);
    T2_recon = T2_recon(:);
    PD_recon = PD_recon(:);
    
    T1_recon((mask<1)|(mask>3))=[];
    T2_recon((mask<1)|(mask>3))=[];
    PD_recon((mask<1)|(mask>3))=[];
    
    T1gt((mask<1)|(mask>3))=[];
    T2gt((mask<1)|(mask>3))=[];
    PDgt((mask<1)|(mask>3))=[];
    
    re1 = mean(abs(T1gt-T1_recon)./T1gt);
    re2 = mean(abs(T2gt-T2_recon)./T2gt);
    re3 = mean(abs(PDgt-abs(PD_recon))./PDgt);
    RE1 = [RE1 re1];
    RE2 = [RE2 re2];
    RE3 = [RE3 re3];
    
    figure(1); scatter(1:niter,RE1); grid on; drawnow;
    figure(2); scatter(1:niter,RE2); grid on; drawnow;
    figure(3); scatter(1:niter,RE3); grid on; drawnow;
end
    
    
        
        
        
        
            
        
        
        
        
    
    
    
    
    
    
    
    
    
    
        
    
    