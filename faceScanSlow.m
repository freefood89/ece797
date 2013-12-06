
[M_E,N_E] = size(E);
[M,N] = size(image);

result = zeros(size(image));
for i=1:M-M_E+1 
    for j=1:N-N_E+1
        patch = image(i:i+M_E-1,j:j+N_E-1);
        patch = patch-mean(patch(:));
        patch = patch/norm(patch(:));
        result(i,j) = patch(:)'*E(:);
    end
end
%%
figure, imagesc(result)