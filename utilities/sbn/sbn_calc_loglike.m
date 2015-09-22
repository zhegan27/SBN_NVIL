
function lb = sbn_calc_loglike(v,parameters)

W=parameters{1}; % M*J
U=parameters{2}; % J*M
b=parameters{3}; % J*1
c=parameters{4}; % M*1
d=parameters{5}; % J*1
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

end        

