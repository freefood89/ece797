function score = faceScan_ren(E, A)

[M_E, N_E] = size(E);
[M_A, N_A] = size(A);
%first compute the integral image
intA = cumsum(cumsum(A,1),2);
intA2 = cumsum(cumsum(A.^2,1),2);
%Now, at each pixel compute the mean
patchmuA = [];
patchnormA = [];
for i = 1:M_A - N_E + 1
    for j = 1:N_A - M_E + 1
        a1 = intA(i,j);
        a2 = intA(i+N_E-1,j);
        a3 = intA(i,j+M_E-1);
        a4 = intA(i+N_E-1,j+M_E-1);
        patchmuA(i,j) = a4 + a1 - a2 - a3;
        a1 = intA2(i,j);
        a2 = intA2(i+N_E-1,j);
        a3 = intA2(i,j+M_E-1);
        a4 = intA2(i+N_E-1,j+M_E-1);
        patchnormA(i,j) = sqrt(a4 + a1 - a2 - a3); 
    end
end
convolved = conv2(double(A), double(fliplr(flipud(E))),'valid');
sumE = sum(E(:));
score = convolved - sumE*patchmuA;%/(M_E*N_E);
score = score./patchnormA;
% figure;
% subplot(131), imagesc(convolved); title('Convolution');
% subplot(132), imagesc(sumE*patchmuA); title('Patch Means');
% subplot(133), imagesc(patchnormA); title('Patch Norms');
end