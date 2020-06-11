function [ Zstar, Z ] = getAnchor( X, p )
    viewNum = size(X,2);   
    N = size(X{1},1);     
    rand('seed',sum(100*clock))
    allIndex = randperm(N);
    anchorIndex = allIndex(1:p);
    for it = 1:viewNum
        Anchor{it} = X{it}(anchorIndex,:);
    end
    fprintf('Nonlinear Anchor Embedding...\n');
    for it = 1:viewNum
        fprintf('The %d-th view Nonlinear Anchor Embeeding...\n',it);
        dist = EuDist2(X{it},Anchor{it},0);
        sigma = mean(min(dist,[],2).^0.5)*2;
        feaVec = exp(-dist/(2*sigma*sigma));  
        Z{it} = bsxfun(@minus, feaVec', mean(feaVec',2));
    end
    
    Zstar = zeros(size(Z{1}));
    for i = 1:viewNum
        Zstar = Zstar + Z{i};
    end
    mn=min(Zstar);
    mx=max(Zstar);
    Zstar=bsxfun(@minus,Zstar,mn);
    feaMaxMin = full(mx-mn);
    Zstar = Zstar./feaMaxMin(ones(size(Zstar,1),1),:);
    feaSum = full((sum(Zstar,1)));
    feaSum = max(feaSum, 1e-12);
    Zstar = Zstar./feaSum(ones(size(Zstar,1),1),:);
    Zstar = Zstar';
end

