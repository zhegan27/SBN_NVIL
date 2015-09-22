function [parameters,logpv_test] = sbn_ascent(v,parameters,opts,vtest)

% Output Results
logpv=[]; logpv_test = [];
[~,N] = size(v);
P=numel(parameters); prevMean = 0; prevVar = 0;

rmsdecay=opts.rmsdecay;
momentum=parameters; 
for p=1:P
    momentum{p}(:)=0;
end

rms=cell(P); g=cell(P);
total_grads = cell(P);

tic
for iter=1:opts.iters

    % Get Gradient
    vB = subsampleData(v,opts.batchSize);
    [grads,~,meanll,varll] = sbn_gradient(vB,parameters,prevMean,prevVar);
    
    prevMean = meanll;
    prevVar = varll;

    if opts.method == 2
        % Update Parameters using RMSprop
        for p=1:P
            if iter == 1
                rms{p} = grads{p}.^2; g{p} = grads{p};
            else
                rms{p} = rmsdecay*rms{p} + (1-rmsdecay)*grads{p}.^2;
                g{p} = rmsdecay*g{p} + (1-rmsdecay)*grads{p};
            end;
            step=grads{p}-opts.penalties*parameters{p};
            step=iter^-opts.decay*opts.stepsize*step;
            step = step./(sqrt(rms{p}-g{p}.^2+1e-4));
            if opts.momentum == 1
                momentum{p}=opts.moment_val*momentum{p}+step;
                step=momentum{p};
            end;
            parameters{p}=parameters{p}+step;
        end
    elseif opts.method == 1
        % Update Parameters using AdaGrad
        for p=1:P
            if iter == 1
                total_grads{p} = grads{p}.^2;
            else
                total_grads{p} = total_grads{p} + grads{p}.^2;
            end;
            step=grads{p}-opts.penalties*parameters{p};
            step=iter^-opts.decay*opts.stepsize*step;
            step = step./(sqrt(total_grads{p})+1e-6);
            if opts.momentum == 1
                momentum{p}=opts.moment_val*momentum{p}+step;
                step=momentum{p};
            end;
            parameters{p}=parameters{p}+step;
        end
    elseif opts.method == 0
        % Update Parameters using SGD
        for p=1:P
            step=grads{p}-opts.penalties*parameters{p};
            step=iter^-opts.decay*opts.stepsize*step;
            if opts.momentum == 1
                momentum{p}=opts.moment_val*momentum{p}+step;
                step=momentum{p};
            end;
            parameters{p}=parameters{p}+step;
        end
    end;

    % Check performance every so often
    if mod(iter,opts.evalInterval)==0
        vB = subsampleData(v,opts.testBatch);
        logpv=[logpv;sbn_calc_loglike(vB,parameters)];
        
        vB = subsampleData(vtest,opts.testBatch);
        logpv_test=[logpv_test;sbn_calc_loglike(vB,parameters)];
                      
        totaltime = toc;
        fprintf('Iter %d: logpv=%4.8f, logpv_test=%4.8f, time=%4.8f\n',...
            iter,logpv(end),logpv_test(end),totaltime);
    end
end
end