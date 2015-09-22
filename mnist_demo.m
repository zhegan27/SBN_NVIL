clear all; clc; close all;
addpath(genpath('.'))

%% Size and parameters for the bouncing-balls data.
load mnist;
[M,N] = size(v);
J=200;

%% Intialize parameters for training
initialParameters{1}=.001*randn(M,J); % W
initialParameters{2}=.001*randn(J,M); % U
initialParameters{3}=zeros(J,1); % b
initialParameters{4}=zeros(M,1); % c
initialParameters{5}=zeros(J,1); % d

% this is the parameters that are used to learn the data-dependent baseline
% using a feed-forward neural network
L = 100;
initialParameters{6}=.001*randn(1,L); % A1
initialParameters{7}=.001*randn(L,M); % A2

%% Training options
opts.iters=1e5; % total iteration number
opts.penalties=1e-4; % weight decay
opts.decay=0; % learning rate decay
opts.momentum = 1; % 1: momentum is used 
opts.evalInterval=200;
opts.moment_val = 0.9;

opts.batchSize = 100;
opts.testBatch = 10000;

% 0: SGD; 1: AdaGrad; 2: RMSprop
opts.method = 2;
opts.stepsize = 1e-3;
opts.rmsdecay = 0.95;

%%
[param,result]=sbn_ascent(v,initialParameters,opts,vtest);


