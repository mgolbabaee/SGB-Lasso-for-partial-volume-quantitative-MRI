function [out] = gen_mag_der_DL(x,nn,mode)
sgn = @(x) max(sign(x),0);

[d1,d2]= size(x); %x nsamp x n_ch0
c = squeeze(nn.NetBw.Layers(1).AverageImage); % Matlab 2019a
% c = squeeze(nn.NetBw.Layers(1).Mean);       % Matlab 2020a
x=x-c.'; %bring back centering

W1 = squeeze(nn.NetBw.Layers(2).Weights); % n_ch0 x n_ch1
B1 = squeeze(nn.NetBw.Layers(2).Bias);
W2 = squeeze(nn.NetBw.Layers(4).Weights); % n_ch1 x n_ch2
B2 = squeeze(nn.NetBw.Layers(4).Bias);

a = x*W1 + repmat(B1.',[d1,1]);
a = max(a,0); % first relu
b = a*W2 + repmat(B2.',[d1,1]);

out.M = double(b);

if mode == 2    
    b = ones(size(b));
    a = sgn(a);
   out.M_dash = zeros(d1,d2,size(b,2));
    for i=1:size(b,2)
        tmp = (b(:,i)* W2(:,i).'); 
        out.M_dash(:,:,i)= (tmp.*a) * W1.';
    end
    out.M_dash = permute(out.M_dash,[1 3 2]);
    out.M_dash = double(out.M_dash);        
end
