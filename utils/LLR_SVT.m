function R = LLR_SVT(X,A,mu,lamda,blocksize,dens,p)
% This function performs local low rank
tmp = X + A./mu;
Mask = zeros(size(tmp,1),size(tmp,2));
R = zeros(size(tmp));
n_block = dens*round(65536/(blocksize.^2));
xc = round(unifrnd(1+(blocksize-1)/2,256-(blocksize-1)/2,1,n_block));
yc = round(unifrnd(1+(blocksize-1)/2,256-(blocksize-1)/2,1,n_block));
for i = 1:n_block
    r = tmp((xc(i)-(blocksize-1)/2):(xc(i)+(blocksize-1)/2), (yc(i)-(blocksize-1)/2):(yc(i)+(blocksize-1)/2), :);
    r = reshape(r,[blocksize.^2, size(tmp,3)]);
    [U,S,V] = svd(r,'econ');
    S = diag(S);
    thresh_r = (lamda/mu).*(S.^(p-1));
    S = S - thresh_r;
    S = S.*(S>0);
    r = U*diag(S)*V';
    r = reshape(r, [blocksize, blocksize, size(tmp,3)]);
    R((xc(i)-(blocksize-1)/2):(xc(i)+(blocksize-1)/2), (yc(i)-(blocksize-1)/2):(yc(i)+(blocksize-1)/2), :) = R((xc(i)-(blocksize-1)/2):(xc(i)+(blocksize-1)/2), (yc(i)-(blocksize-1)/2):(yc(i)+(blocksize-1)/2), :) + r;
    Mask((xc(i)-(blocksize-1)/2):(xc(i)+(blocksize-1)/2), (yc(i)-(blocksize-1)/2):(yc(i)+(blocksize-1)/2)) = Mask((xc(i)-(blocksize-1)/2):(xc(i)+(blocksize-1)/2), (yc(i)-(blocksize-1)/2):(yc(i)+(blocksize-1)/2)) + ones(blocksize, blocksize);
end
R(R==0)=tmp(R==0);
Mask(Mask==0)=1;
R = R./repmat(Mask,[1 1 size(tmp,3)]);
end

