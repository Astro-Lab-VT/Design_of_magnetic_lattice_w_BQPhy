clear;clc;
load data

%% Parameters
maxGen = 100;
nVars = 225;
numChromosomes = 100;
deltaTheta = 0.45;
penaltyFactor = 1;
initialPenalty = 1;
maxOuterIter = 1;

%% Optimization
tic
objFun = @(opt_space) FreeEnergy_stochastic_same_exact_wrapper_CL(opt_space, x, y);
[BestDesign, BestFitness, GenFitness] = BQPhy_Optimiser(objFun, ...
    'numChromosomes', numChromosomes, ...
    'dimension', nVars, ...
    'deltaTheta', deltaTheta, ...
    'maxGen', maxGen, ...
    'penaltyFactor', penaltyFactor, ...
    'initialPenalty', initialPenalty, ...
    'maxOuterIter', maxOuterIter, ...
    'typeOfOptimization','BINARY');

%% Fitness Evaluation
filtered = GenFitness(1);  % Always include the first element
current_min = GenFitness(1);

for i = 2:length(GenFitness)
    if GenFitness(i) < current_min
        filtered(i) = GenFitness(i);
        current_min = GenFitness(i);
    else
        filtered(i) = current_min;
    end
end
toc
% Plot original and filtered
figure();
plot(filtered, '-o', 'DisplayName', 'Original'); hold on;
legend;
xlabel('Generation Id');
ylabel('Fitness value');
title('Filter to Keep Only New Minimums');
grid on;
