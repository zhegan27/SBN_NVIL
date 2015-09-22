
function [gradient,lb,meanll,varll] = sbn_gradient(v,parameters,prevMean,prevVar)

alpha = 0.8;

W=parameters{1}; % M*J
U=parameters{2}; % J*M
b=parameters{3}; % J*1
c=parameters{4}; % M*1
d=parameters{5}; % J*1

A1 = parameters{6}; % 1*L
A2 = parameters{7}; % L*M
N = size(v,2);

%% calculate lower bound
term2 = bsxfun(@plus,U*v,d); % J*N
h=double(sigmoid(term2)>rand(size(term2))); % J*N

term1 = bsxfun(@plus,W*h,c); % M*N
logprior = b'*h - sum(log(1+exp(b))); % 1*N 
loglike = sum(term1.*v-log(1+exp(term1))); % 1*N
logpost = sum(term2.*h-log(1+exp(term2))); % 1*N
ll = logprior + loglike - logpost; % 1*N
lb = mean(ll);

ll = ll - A1*tanh(A2*v);

if prevMean == 0 && prevVar == 0
    meanll = mean(ll);
    varll = var(ll);
else
    meanll = alpha*prevMean + (1-alpha)*mean(ll); 
    varll = alpha*prevVar + (1-alpha)*var(ll);
end;

ll = (ll-meanll)./max(1,sqrt(varll));

%% gradient information
grads.W = (v-sigmoid(term1))*h'/N; % M*J
grads.c = sum(v-sigmoid(term1),2)/N; % M*1
grads.b = mean(h,2)-sigmoid(b); % J*1
grads.U = (bsxfun(@times,h-sigmoid(term2),ll))*v'/N; % J*M
grads.d = sum(bsxfun(@times,h-sigmoid(term2),ll),2)/N; % J*1

grads.A1 = sum(bsxfun(@times,tanh(A2*v),ll),2)'/N; % 1*L 
tmp1 = bsxfun(@times,1-tanh(A2*v).^2,ll); %L*T
tmp2 = bsxfun(@times,tmp1,A1'); % L*T
grads.A2 = tmp2*v'/N;

%% collection
gradient{1} = grads.W;
gradient{2} = grads.U;
gradient{3} = grads.b;
gradient{4} = grads.c;
gradient{5} = grads.d;   

gradient{6} = grads.A1; 
gradient{7} = grads.A2; 

end        
