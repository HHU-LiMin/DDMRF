function Dict = KSVD_CDL_kt(...
    Data,...  % a cell containing training data from multiple contrast images.
    param)

% MODIFIED K-SVD IMPLEMENTATION THAT WORKS FOR Coupled Dictionary Learning.
% TRAINING CAN HAVE BOTH SPARSITY LEVEL AND ERROR THRESHOLD TOGETHER.

% =========================================================================
%                          K-SVD algorithm
% =========================================================================
% The K-SVD algorithm finds a dictionary for linear representation of
% signals. Given a set of signals, it searches for the best dictionary that
% can sparsely represent each signal. Detailed discussion on the algorithm
% and possible applications can be found in "The K-SVD: An Algorithm for 
% Designing of Overcomplete Dictionaries for Sparse Representation", written
% by M. Aharon, M. Elad, and A.M. Bruckstein and appeared in the IEEE Trans. 
% On Signal Processing, Vol. 54, no. 11, pp. 4311-4322, November 2006. 
% =========================================================================
% INPUT ARGUMENTS:
% Data                         Cell£¬contains R n#N matrix that contins N signals, each of dimension n. 
% param                        structure that includes all required parameters for the K-SVD execution, required fields are:
%    n                         Variable, dimension of a dictionary element
%    K                         Variable, number of coupled dictionary elements to train
%    R                         Variable, number of coupled MR contrast images
%    numIter_KSVD              Variable, number of iterations to perform.
%    iniDictionary_coupled     Cell, contains R n#K coupled dictionaries.
%    s                         Variable, maximum coefficients to use in OMP coefficient calculations.
%    thr                       Variable, Error threshold used within sparse coding (along with fixed sparsity level)
% =========================================================================
% OUTPUT ARGUMENTS:
%  Dict                        The extracted dictionary of size nX(param.K).
% =========================================================================


n = param.n;
K = param.K;
R = param.R;
numIter_KSVD = param.numIter_KSVD;
iniDictionary_coupled = param.iniDictionary_coupled;
s = param.s;
thr = param.thr;    

% prepare coupled training data
TrainingData = [];
for r = 1:R
    TrainingData = [TrainingData; Data{r}];
end

% prepare initalization dictionary
Dict = [];
for r = 1:R
    Dict = [Dict; iniDictionary_coupled{r}];
end

% normalize the dictionary
Dict = Dict*diag(1./sqrt(sum((conj(Dict)).*Dict)));

%% the coupled dictionary learning algorithm starts here.
for iterNum = 1:numIter_KSVD
    % global sparse coding
    CoefMatrix = OMPerrn(Dict,TrainingData,thr,s);
    
    % update dictionaries
    rPerm = randperm(K);
    replacedVectorCounter = 0;
    for j = rPerm
        [betterDictionaryElement,CoefMatrix,addedNewVector] = I_findBetterDictionaryElement(TrainingData,...
            Dict,j,CoefMatrix);
        Dict(:,j) = betterDictionaryElement;
        replacedVectorCounter = replacedVectorCounter + addedNewVector;
    end
    Dict = I_clearDictionary(Dict,CoefMatrix,TrainingData);
%     CoefMatrix(1:K,:) = CoefMatrix_coupled;
end

 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  findBetterDictionaryElement
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [betterDictionaryElement,CoefMatrix,NewVectorAdded] = I_findBetterDictionaryElement(Data,Dictionary,j,CoefMatrix)
relevantDataIndices = find(abs(CoefMatrix(j,:))); % the data indices that uses the j'th dictionary element.
if (length(relevantDataIndices)<1) %(length(relevantDataIndices)==0)
    ErrorMat = Data-Dictionary*CoefMatrix;
    ErrorNormVec = sum((abs(ErrorMat)).^2);
    [d,i] = max(ErrorNormVec);
    betterDictionaryElement = Data(:,i);%ErrorMat(:,i); %
    betterDictionaryElement = betterDictionaryElement./sqrt(betterDictionaryElement'*betterDictionaryElement);
    %betterDictionaryElement = betterDictionaryElement.*sign(betterDictionaryElement(1));
    CoefMatrix(j,:) = 0;
    NewVectorAdded = 1;
    return;
end
NewVectorAdded = 0;
tmpCoefMatrix = CoefMatrix(:,relevantDataIndices); 
tmpCoefMatrix(j,:) = 0;% the coeffitients of the element we now improve are not relevant.
errors =(Data(:,relevantDataIndices) - Dictionary*tmpCoefMatrix); % vector of errors that we want to minimize with the new element
% % the better dictionary element and the values of beta are found using svd.
% % This is because we would like to minimize || errors - beta*element ||_F^2. 
% % that is, to approximate the matrix 'errors' with a one-rank matrix. This
% % is done using the largest singular value.
[betterDictionaryElement,singularValue,betaVector] = svds(errors,1);
CoefMatrix(j,relevantDataIndices) = singularValue*betaVector';% *signOfFirstElem


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  I_clearDictionary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Dictionary = I_clearDictionary(Dictionary,CoefMatrix,Data)
T2 = 0.99;
T1 = 10;
K=size(Dictionary,2);
Er=sum((abs(Data-Dictionary*CoefMatrix)).^2,1); % remove identical atoms
G=abs(Dictionary'*Dictionary); G = G-diag(diag(G));
for jj=1:K
    if max(G(jj,:))>T2 | length(find(abs(CoefMatrix(jj,:))>1e-7))<=T1 
        [val,pos]=max(Er);
        Er(pos(1))=0;
        Dictionary(:,jj)=Data(:,pos(1))/norm(Data(:,pos(1)));
        G=abs(Dictionary'*Dictionary); G = G-diag(diag(G));
    end
end


