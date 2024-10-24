function  X_nuc = florAlg( Y,D,N,L,r,th )

% [ X_estimated_flor, E_estimated_flor] = florAlg( Y,D,N,L,X,r,th );
% Y是加噪声的欠采样k空间数据 
% D是字典 
% 图像的大小 NxN
% 序列的长度 L
% X为真实值(这里的X在原代码中代表的是全采样数据 相当于我们论文中的GroundTruth的作用)
% r 是指在软阈值中要取得的前r个特征值
% th = 5; th是软阈值法中要用到的阈值大小


%This function calculates FLOR algorithm
Y_sticks = reshape(Y,N*N,L);  % 对k空间数据 Y 进行重新排列
base = orth(D.');
X_nuc = minNuc( Y_sticks,base.' ,r,th);
X_nuc = reshape(X_nuc,N,N,L);
end

