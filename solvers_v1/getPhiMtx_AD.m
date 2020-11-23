function M = getPhiMtx_AD(x,Phi0,L,mode)
nPhi=[];
d1Phi=[];
d2Phi=[];

P = Phi0(x,mode);
P0 = (P.M).';
nPhi =  P0.* repmat(1./sqrt(sum(abs(P0).^2,1)), [L,1] );

if mode ==2    
    D = P.M_dash;
    D1 = D(:,:,1).';
    D2 = D(:,:,2).';
    
    ndPhi = @(D,P0) D.* repmat(1./sqrt(sum(abs(P0).^2,1)), [L,1] ) ...
        - P0.* repmat(sum(P0.*D,1)./(sum(abs(P0).^2,1)).^(3/2), [L,1] ) ;
    
    d1Phi = ndPhi(D1,P0);
    d2Phi = ndPhi(D2,P0);
end
M ={nPhi,d1Phi,d2Phi};
end
