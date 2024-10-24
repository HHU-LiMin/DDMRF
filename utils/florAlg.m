function  X_nuc = florAlg( Y,D,N,L,r,th )

% [ X_estimated_flor, E_estimated_flor] = florAlg( Y,D,N,L,X,r,th );
% Y�Ǽ�������Ƿ����k�ռ����� 
% D���ֵ� 
% ͼ��Ĵ�С NxN
% ���еĳ��� L
% XΪ��ʵֵ(�����X��ԭ�����д������ȫ�������� �൱�����������е�GroundTruth������)
% r ��ָ������ֵ��Ҫȡ�õ�ǰr������ֵ
% th = 5; th������ֵ����Ҫ�õ�����ֵ��С


%This function calculates FLOR algorithm
Y_sticks = reshape(Y,N*N,L);  % ��k�ռ����� Y ������������
base = orth(D.');
X_nuc = minNuc( Y_sticks,base.' ,r,th);
X_nuc = reshape(X_nuc,N,N,L);
end

