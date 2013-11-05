%% 18797 Boosting Based Face Detector

if(0) 
    ece797_hw2_parameterize; 
end
clear all;
%% Load Image Vectors

load('trainingVectors.mat');
[K, numF] = size(Wf);
numNF = size(Wnf,2);

%% Constant Definition and Allocation

iter= 20;

ht = zeros(1,iter); % classifier threshold
st = zeros(1,iter); % classifier direction
at = zeros(1,iter); % classifier weight
misst = zeros(1,iter); % classifier mistakes
h_dim = zeros(1,iter); % classifier component

set = [Wf Wnf];
sol = [ones(1,numF) -ones(1,numNF)];
thres = zeros(1,1+numF+numNF);
err = zeros(size(thres));
w_i = ones(1,numF+numNF)/(numF+numNF);
Dt = w_i;
Htrain=0;
for q=1:iter
    side = zeros(1,K);
    h = zeros(1,K);
    for p=1:K
        setp = set(p,:);
        E = sort(setp);
        diffE = diff(E);
        thres(2:end-1) = E(2:end) - diffE/2;
        thres(1) = thres(2)-diffE(1);
        thres(end) = thres(end-1)+diffE(end);
        for i=1:length(thres)
            Hx = 2*(setp > thres(i))-1;
            err(i) = 0.5*sum(Dt.*(1-Hx.*sol));
        end
        absErr = abs(err-.5);
        m = max(absErr)==absErr;
        tmp = thres(m);
        h(p) = tmp(1);
        
        m_aboveH = (setp>h(p));
        m_belowH = ~m_aboveH;
        m_face = sol==1;
        m_nonface = ~m_face;
        c1 = sum(Dt((m_face&m_belowH)|(m_nonface&m_aboveH)));
        side(p) = 2*(c1<0.5)-1;
        if(side(p)==1)
            e(p) = c1;
        else
            e(p) = 1-c1;
        end
        %     figure, plot(thresholds, [e_t; abs(e_t-0.5)]);
    end
    h_dim(q) = min(find(e==min(e))); % eigface with the best classifier
    seth = set(h_dim(q),:);
    st(q) = side(h_dim(q));
    ht(q) = h(h_dim(q)); % the best weak classifier
    if(side(h_dim(q))==1)
        Hx = 2*(seth > ht(q))-1; % faces are above threshold
    else
        Hx = 2*(seth < ht(q))-1; % faces are below threshold
    end
    et = 0.5*sum(Dt.*(1-Hx.*sol)); % error of best classfier
    at(q) = log((1-et)/et)/2;
%     [h_dim(pp) ht(pp) st(pp) at(pp)]
    Dt = Dt.*exp(-at(q).*sol.*Hx);
    Dt = Dt/sum(Dt);
    
    Htrain = Htrain+at(q)*Hx;
    tmp = sign(Htrain);
    misst(q) = sum(logical((1-sol.*tmp)/2))/(numF+numNF)
q
end
%% Displaying Results
% Display multiple classifiers
figure;
plot(set(1,sol==1),set(2,sol==1),'b.'); hold on;
plot(set(1,sol==-1),set(2,sol==-1),'r.');

for p=1:iter
    if (h_dim(p)==1)
        plot([ht(p) ht(p)],[-1 1]);
    elseif (h_dim(p)==2)
        plot([-1 1],[ht(p) ht(p)]);
    end
end
Htrain = sign(Htrain);
missed = logical((1-sol.*Htrain)/2);
plot(set(1,missed),set(2,missed),'go');
hold off;
xlabel('E_1'); ylabel('E_2'); title('training set');
legend('Face','Not Face','ht','Error');

%%

load('testingVectors.mat');
set = [Wf Wnf];
[K, numF] = size(Wf);
numNF = size(Wnf,2);
sol = [ones(1,numF) -ones(1,numNF)];
Htest=0;
for p=1:length(ht)
    if(side(h_dim(p))==1)
        Hx = 2*(set(h_dim(p),:) > ht(p))-1; % faces are above threshold
    else
        Hx = 2*(set(h_dim(p),:) < ht(p))-1; % faces are below threshold
    end
    Htest = Htest + at(p)*Hx;
    class = sign(Htest);
    misst(p) = sum(logical((1-sol.*class)/2))/(numF+numNF);
end

