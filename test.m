face1 = [.3 .5 .7 .6];
nface1= [.2 -.8 .4 .2];
face2 = [-.6 -.5 -.1 -.4];
nface2= [.4 -.1 -.9 .5];
set = [face1 nface1];
weight = ones(1,8)/length(set);
class = [ones(1,4) -ones(1,4)];
[E ind] = sort(set);
thres = E(2:end) - 0.5*diff(E);
err = zeros(size(thres));
numofF = zeros(size(thres));
numofNF = zeros(size(thres));
for i=1:length(thres)
    Hx = 2*(set > thres(i))-1;
    err(i) = 0.5*sum(weight.*(1-Hx.*class));
    numofF(i) = sum(Wf(p,:) > thres(i));
    numofNF(i) = sum(Wnf(p,:) > thres(i));
end
mins = thres(min(err)==err);
h(p) = mins(1);

figure; subplot(121);
plot(face1,face2,'bo'); hold on;
plot(nface1,nface2,'ro');
plot([h(p) h(p)],[-.4 .3])
xlabel('E_1'); ylabel('E_2');
hold off;

subplot(122), plot(thres,err);
