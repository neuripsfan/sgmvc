clear;
clc;
addpath('./Utility');
addpath('./Dataset');

%---------------------- Parameters ----------------------------------------
opt = [];
opt.p = 1000;           % number of anchors
opt.MaxIter = 7;       % 5 iterations are okay, but better results for 10
opt.innerMax = 5;       % maximum iterations DPLM for subproblem of B and C
opt.r = 5;              % r is the power of alpha_i
opt.L = 128;            % Hashing code length
opt.beta = 0.003;       % Hyper-para beta
opt.gamma = 1e-5;       % Hyper-para gamma
opt.lambda = 5e-5;      % Hyper-para lambda
%----------------------End of Parameters ----------------------------------



%----------------------Load Dataset----------------------------------------
%load CALTECH101.mat;  

load DIGIT.mat;        %load dataset
load DIGITopt.mat;     %load parameters

%----------------------End of Load Dataset---------------------------------



%----------------------Anchor Embedding------------------------------------
[Zstar, Z] = getAnchor(X,opt.p);
clear X;
%------------------ ---End of Anchor Embedding-----------------------------



%----------------------Main SGMVC------------------------------------------
SGMVC(Z, Zstar ,Y,opt);
%----------------------End of Main SGMVC-----------------------------------
