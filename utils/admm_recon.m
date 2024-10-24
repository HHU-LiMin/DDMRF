function [qMaps, PD, x, r] = admm_recon(E, data, Dic, ADMM_iter, cg_iter, mu1, mu2, lambda, P, verbose)
% Reconstructs quantitative maps from k-space data by alternately
% solving the inverse imaging problem, constraint to be close to the latest
% dictionary fit, and fitting the series of images to the dictionary.
%
% [qMaps, PD, x]    = admm_recon(E, data, Dic)
% [qMaps, PD, x]    = admm_recon(E, data, Dic, ADMM_iter)
% [qMaps, PD, x]    = admm_recon(E, data, Dic, ADMM_iter, cg_iter)
% [qMaps, PD, x]    = admm_recon(E, data, Dic, ADMM_iter, cg_iter, mu1)
% [qMaps, PD, x]    = admm_recon(E, data, Dic, ADMM_iter, cg_iter, mu1, mu2, lambda)
% [qMaps, PD, x]    = admm_recon(E, data, Dic, ADMM_iter, cg_iter, mu1, mu2, lambda, P)
% [qMaps, PD, x]    = admm_recon(E, data, Dic, ADMM_iter, cg_iter, mu1, mu2, lambda, P, verbose)
% [qMaps, PD, x, r] = admm_recon(___)
%
% Input:
%   E         =  Imaging operator (use LR_nuFFT_operator provided by this
%                toolbox. It can be used for a low rank approximation of the
%                time series, but also for a time frame by time frame
%                reconstruction.
%   data      in [n_samples*nt ncoils]
%                k-space data to be reconstructed. The first dimension
%                represents the readout of all time frames concatted and
%                the second dimension is allows multi-coil data, if E
%                includes the corresponding sensitivity maps.
%   Dic       =  Dictionary struct (see MRF_dictionary.m for details)
%   ADMM_iter =  number of ADMM iterations (default = 10)
%   cg_iter   =  number of conjugate gradient iterations in each ADMM
%                iteration (default = 20)
%   mu1       =  ADMM coupling parameter (dictionary) (default = 1.26e-6,
%                but needs to be changed very likely)
%   mu2       =  ADMM coupling parameter to the spatial regularization
%                term. Has only an effect, if lambda>0 (default = .25)
%   lambda    =  Regularization parameter (default = 0, which results in no
%                spatial regularization)
%   P         =  operator that transforms the images into the space,
%                in which an l21-norm penalty is applied. Has only an
%                effect if lambda>0. Default = 1 (penalty in the image
%                space).
%                Examples:
%                P = wavelet_operator([nx ny nz], 3, 'db2');
%                P = finite_difference_operator([1 2 3]);
%   verbose   =  0 for no output, 1 for plotting the images and in each
%                iteration and give only one output per ADMM iteration in
%                the commandline and 2 for also print the residal of the
%                CG in each CG iteration.
%
%
%
% Output:
%   qMaps = Maps of quantities contained in D.lookup_table
%   PD    = Proton density retrived from the correlation
%   x     = Low rank - or time-series of images
%   r     = residual after all ADMM steps. Use only when you really want to
%           know it since it requires and additional nuFFT operation
%
% For more details, please refer to
%   J. Asslaender, M.A. Cloos, F. Knoll, D.K. Sodickson, J.Hennig and
%   R. Lattanzi, Low Rank Alternating Direction Method of Multipliers
%   Reconstruction for MR Fingerprinting  Magn. Reson. Med., epub
%   ahead of print, 2016.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Jakob Asslaender, August 2016
% New York University School of Medicine, Center for Biomedical Imaging
% University Medical Center Freiburg, Medical Physics
% jakob.asslaender@nyumc.org
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Manage the input...
% If E is a LR_nuFFT_operator, size(E,2) returns [nx ny (nz) R(nt)]
recon_dim = size(E,2);

if nargin < 4 || isempty(ADMM_iter)
    ADMM_iter = 10;
end
if nargin < 5 || isempty(cg_iter)
    cg_iter = 20;
end
if nargin < 6 || isempty(mu1)
    mu1 = 1.26e-3;
end
if nargin < 7 || isempty(mu2)
    mu2 = .25;
end
if nargin < 8 || isempty(lambda)
    lambda = 0;
end
if nargin < 9 || isempty(P)
    P = 1;
end
if nargin < 10 || isempty(verbose)
    verbose = 1;
end

% ploting stuff
persistent h1 h2 h3 h4 h5
for param = 1:size(Dic.lookup_table,2)
    eval(['persistent h', num2str(param+5)]);
end

%% Initaialize
backprojection = E' * data;
x = 0;
y = zeros(recon_dim);
D = zeros(recon_dim);
if lambda > 0
    Px = zeros(size(P * backprojection));
    z = Px;
    G = Px;
end

r = zeros(1,ADMM_iter);

RE1 = [];
RE2 = [];
RE3 = [];
RD = [] ;
%% ADMM Iterations
for j=1:ADMM_iter
    tic;
    
    %% update x
    b = backprojection - mu1 * y + mu1 * D .* repmat(sum(conj(D) .* y, length(recon_dim)), [ones(1,length(recon_dim)-1) recon_dim(end)]);
    if lambda > 0
        b = b + mu2 * (P' * (G - z));
        % if j==0, mu1 = 0 in order to realize (DDh)_0 = 1
        f = @(x) E'*(E*x) + (mu1 * (j>1)) * (x - D .* repmat(sum(conj(D) .* x, length(recon_dim)), [ones(1,length(recon_dim)-1) recon_dim(end)])) + (mu2 * (j>1)) * (P' * (P * x));
        %         f = @(x) E'*(E*x) + (mu1 * (j>1)) * (x - D .* repmat(sum(conj(D) .* x, length(recon_dim)), [ones(1,length(recon_dim)-1) recon_dim(end)])) + mu2 * (P' * (P * x));
    else
        % if j==0, mu1 = 0 in order to realize (DDh)_0 = 1
        f = @(x) E'*(E*x) + (mu1 * (j>1)) * (x - D .* repmat(sum(conj(D) .* x, length(recon_dim)), [ones(1,length(recon_dim)-1) recon_dim(end)]));
    end
    x = conjugate_gradient(f,b,1e-6,cg_iter,x,verbose);
    
    
    %% update D and y (and claculate PD and qMaps for the output)
    x  = reshape(x, [prod(recon_dim(1:end-1)), recon_dim(end)]);
    y  = reshape(y, [prod(recon_dim(1:end-1)), recon_dim(end)]);
    D  = reshape(D, [prod(recon_dim(1:end-1)), recon_dim(end)]);
    
    for q=size(x,1):-1:1
        Dx = x(q,:) * Dic.magnetization;
        Dy = y(q,:) * Dic.magnetization;
        [~,idx(q,1)] = max(2*real(Dx.*conj(Dy)) + abs(Dx).^2, [], 2);
    end
    
    DDhx_old = reshape(D .* repmat(sum(conj(D) .* x ,2), [1 recon_dim(end)]), recon_dim);
    
    D = double(Dic.magnetization(:,idx)).';
    Dhx = sum(conj(D) .* x ,2);
    PD = Dhx ./ Dic.normalization(idx).';
    qMaps = Dic.lookup_table(idx,:);
    
    x   = reshape(x,    recon_dim);
    y   = reshape(y,    recon_dim);
    D   = reshape(D,    recon_dim);
    Dhx = reshape(Dhx,  recon_dim(1:end-1));
    PD  = reshape(PD,   recon_dim(1:end-1));
    qMaps = reshape(qMaps, [recon_dim(1:end-1), size(Dic.lookup_table,2)]);
    
    DDhx = D .* repmat(Dhx, [ones(1,length(recon_dim)-1) recon_dim(end)]);
    y = y + x - DDhx;
    
    % Dynamic update of mu1 according to Boyd et al. 2011
    sd = l2_norm(mu1 * (DDhx - DDhx_old));
    rd = l2_norm(x - DDhx);
%         if rd > 10 * sd
%             mu1 = 2*mu1;
%             y = y/2;
%         elseif sd > 10 * rd
%             mu1 = mu1/2;
%             y = 2*y;
%         end
%     
    %% update G and z
    if lambda > 0
        G_old = G;
        Px = P * x;
        G = Px + z;
        Tl2 = l2_norm(G, length(size(G)));
        G = G ./ repmat(Tl2, [ones(1, length(size(G))-1) recon_dim(end)]);
        G = G .* repmat(max(Tl2 - lambda/mu2, 0), [ones(1, length(size(G))-1) recon_dim(end)]);
        G(isnan(G)) = 0;
        z = z + Px - G;
        
        % Dynamic update of mu2 according to Boyd et al. 2011
        rs = l2_norm(Px - G);
        ss = l2_norm(mu2 * (P' * (G - G_old)));
        if rs > 10 * ss
            mu2 = 2*mu2;
            z = z/2;
        elseif ss > 10 * rs
            mu2 = mu2/2;
            z = 2*z;
        end
    end
    

    %% Below here is just plotting stuff...
      if verbose == 1
        % display D*Dh*x and (x-D*Dh*x)
        if (isempty(h1) || ~ishandle(h1)), h1 = figure; end; set(0,'CurrentFigure',h1);
        imagesc34d(abs(    DDhx),0); title([      'D*Dh*x - iteration = ', num2str(j)]); colorbar; colormap gray; axis off;
        if (isempty(h2) || ~ishandle(h2)), h2 = figure; end; set(0,'CurrentFigure',h2);
        imagesc34d(abs(x - DDhx),0); title(['(x - D*Dh*x) - iteration = ', num2str(j)]); colorbar; colormap gray; axis off;
        
     
      end
    
    
    %% calculate MSE
    load('E:\MRFcode\MRFRecon_Toolbox\Code_MATLAB\Brain.mat');
    
    mask = mask(:);
    T1gt = T1gt(:);
    T2gt = T2gt(:);
    PDgt = PDgt(:);
    
    T1_recon = qMaps(:,:,1);
    T2_recon = qMaps(:,:,2);
    PD_recon = PD;
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
    
    
    %º∆À„≤–≤Ó
    rd = l2_norm(x - DDhx);
    RD = [RD rd];
    
    figure(3); scatter(1:j,RD); grid on; drawnow;
    figure(4); scatter(1:j,RE1); grid on; drawnow;
    figure(5); scatter(1:j,RE2); grid on; drawnow;
    figure(6); scatter(1:j,RE3); grid on; drawnow;
end
end