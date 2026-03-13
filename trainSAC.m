%TRAINSAC Train a Soft Actor-Critic agent for RAN slicing control.
%   This script creates the custom environment, configures an SAC agent,
%   sets training options with periodic checkpointing, and launches
%   training using MATLAB Reinforcement Learning Toolbox.
%
%   Outputs in base workspace:
%   - trainingStats: Training statistics returned by train().
%
%   Saved artifacts:
%   - Agent checkpoints under Config.train_save_dir
%   - trainingStats.mat under Config.train_save_dir

clear;
clc;

fprintf('\n===============================================================\n');
fprintf('trainSAC.m launched at %s\n', char(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss")));
fprintf('===============================================================\n');

rng(Config.random_seed);
fprintf('Random seed fixed to %d.\n', Config.random_seed);

env = RANSlicingEnv();
initialObservation = reset(env);

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
agent = rlSACAgent(obsInfo, actInfo);

agentOpts = agent.AgentOptions;
if isprop(agentOpts, "MiniBatchSize")
    agentOpts.MiniBatchSize = Config.sac_minibatch_size;
end

if isprop(agentOpts, "NumStepsPerUpdate")
    agentOpts.NumStepsPerUpdate = 10;
end

if isprop(agentOpts, "NumWarmStartSteps")
    agentOpts.NumWarmStartSteps = min(100, Config.Max_Episode_Steps);
end

if isprop(agentOpts, "ActorOptimizerOptions")
    if isprop(agentOpts.ActorOptimizerOptions, "LearnRate")
        agentOpts.ActorOptimizerOptions.LearnRate = Config.sac_actor_learn_rate;
    end
    if isprop(agentOpts.ActorOptimizerOptions, "GradientThreshold")
        agentOpts.ActorOptimizerOptions.GradientThreshold = 1.0;
    end
elseif isprop(agentOpts, "PolicyOptimizerOptions")
    if isprop(agentOpts.PolicyOptimizerOptions, "LearnRate")
        agentOpts.PolicyOptimizerOptions.LearnRate = Config.sac_actor_learn_rate;
    end
    if isprop(agentOpts.PolicyOptimizerOptions, "GradientThreshold")
        agentOpts.PolicyOptimizerOptions.GradientThreshold = 1.0;
    end
end

if isprop(agentOpts, "CriticOptimizerOptions")
    criticOpts = agentOpts.CriticOptimizerOptions;
    if iscell(criticOpts)
        for idx = 1:numel(criticOpts)
            if isprop(criticOpts{idx}, "LearnRate")
                criticOpts{idx}.LearnRate = Config.sac_critic_learn_rate;
            end
            if isprop(criticOpts{idx}, "GradientThreshold")
                criticOpts{idx}.GradientThreshold = 1.0;
            end
        end
        agentOpts.CriticOptimizerOptions = criticOpts;
    else
        for idx = 1:numel(criticOpts)
            if isprop(criticOpts(idx), "LearnRate")
                criticOpts(idx).LearnRate = Config.sac_critic_learn_rate;
            end
            if isprop(criticOpts(idx), "GradientThreshold")
                criticOpts(idx).GradientThreshold = 1.0;
            end
        end
        agentOpts.CriticOptimizerOptions = criticOpts;
    end
end

agent.AgentOptions = agentOpts;

saveDir = fullfile(pwd, char(Config.train_save_dir));
if ~exist(saveDir, "dir")
    mkdir(saveDir);
end

trainOpts = rlTrainingOptions( ...
    "MaxEpisodes", Config.train_max_episodes, ...
    "MaxStepsPerEpisode", Config.Max_Episode_Steps, ...
    "Plots", "training-progress", ...
    "StopTrainingCriteria", "None", ...
    "SaveAgentCriteria", "EpisodeCount", ...
    "SaveAgentValue", Config.train_save_every_episodes, ...
    "SaveAgentDirectory", saveDir);

if isprop(trainOpts, "Verbose")
    trainOpts.Verbose = true;
end

printSectionHeader("TRAINING HYPERPARAMETERS");
fprintf('Save directory              : %s\n', saveDir);
fprintf('Max episodes                : %d\n', Config.train_max_episodes);
fprintf('Max steps per episode       : %d\n', Config.Max_Episode_Steps);
fprintf('Checkpoint interval         : every %d episodes\n', Config.train_save_every_episodes);
fprintf('Observation dimension       : %s\n', mat2str(obsInfo.Dimension));
fprintf('Action dimension            : %s\n', mat2str(actInfo.Dimension));
fprintf('Action lower bounds         : %s\n', formatNumericVector(actInfo.LowerLimit, '%.2f'));
fprintf('Action upper bounds         : %s\n', formatNumericVector(actInfo.UpperLimit, '%.2f'));
fprintf('SAC minibatch size          : %d\n', readOptionValue(agent.AgentOptions, "MiniBatchSize"));
fprintf('SAC warm start steps        : %d\n', readOptionValue(agent.AgentOptions, "NumWarmStartSteps"));
fprintf('SAC steps per update        : %d\n', readOptionValue(agent.AgentOptions, "NumStepsPerUpdate"));
fprintf('Actor learning rate         : %.3g\n', readOptimizerValue(agent.AgentOptions, ...
    ["ActorOptimizerOptions", "PolicyOptimizerOptions"], "LearnRate"));
fprintf('Actor gradient threshold    : %.3f\n', readOptimizerValue(agent.AgentOptions, ...
    ["ActorOptimizerOptions", "PolicyOptimizerOptions"], "GradientThreshold"));
fprintf('Critic learning rate        : %.3g\n', readOptimizerValue(agent.AgentOptions, ...
    "CriticOptimizerOptions", "LearnRate"));
fprintf('Critic gradient threshold   : %.3f\n', readOptimizerValue(agent.AgentOptions, ...
    "CriticOptimizerOptions", "GradientThreshold"));
fprintf('URLLC arrival lambda        : %.4f\n', Config.lambda_urllc);
fprintf('URLLC packet bits           : %.1f\n', Config.D_one);
fprintf('URLLC delay target          : %.4f ms\n', Config.tau_req * 1e3);
fprintf('eMBB arrival lambda         : %.6f\n', Config.lambda_embb);
fprintf('eMBB Pareto xm bits         : %.1f\n', Config.embb_xm_bits);
fprintf('eMBB Pareto alpha           : %.3f\n', Config.embb_alpha);
fprintf('eMBB penalty scale          : %.3g\n', Config.embb_penalty_scale);
fprintf('Grouping config             : N_g=%d | Q_levels=%d | phi_th=%.1f\n', ...
    Config.N_g, Config.Q_levels, Config.phi_th);
fprintf('Reward weights              : omega=[%.2f, %.2f, %.2f]\n', ...
    Config.omega_1, Config.omega_2, Config.omega_3);
fprintf('SIC thresholds              : threshold=%.2f | min_superposition=%.2f\n', ...
    Config.sic_threshold, Config.sic_min_superposition);
fprintf('Radio config                : RBs=%d | BW/RB=%.0f kHz | mini-slots=%d | slot=%.3f ms\n', ...
    Config.B_RBs, Config.W / 1e3, Config.M_mini_slots, Config.Slot_duration * 1e3);
fprintf('Link budget                 : Tx=%.1f dBm | Noise=%.1f dBm | PathLoss=%.1f dB\n', ...
    Config.Tx_Power_dBm, Config.Noise_Power, Config.Path_Loss_dB);

printSectionHeader("INITIAL ENVIRONMENT SNAPSHOT");
fprintf('Initial observation         : %s\n', formatNumericVector(initialObservation, '%.4f'));
fprintf('Initial mini-slot index     : %d / %d\n', env.MiniSlotIndex, Config.M_mini_slots);
fprintf('Initial URLLC queue sum     : %.1f bits\n', sum(env.URLLCGroupQueues));
fprintf('Initial eMBB queue sum      : %.1f bits\n', sum(env.eMBBGroupQueues));
fprintf('URLLC channel gain stats    : min=%.3e | mean=%.3e | max=%.3e\n', ...
    min(env.URLLCChannelGain), mean(env.URLLCChannelGain), max(env.URLLCChannelGain));
fprintf('eMBB channel gain stats     : min=%.3e | mean=%.3e | max=%.3e\n', ...
    min(env.eMBBChannelGain), mean(env.eMBBChannelGain), max(env.eMBBChannelGain));
fprintf('Current RB group counts     : %s\n', formatNumericVector(currentGroupCounts(env), '%.0f'));

printSectionHeader("OPTIMIZER OBJECTS");
disp("=== DEBUG: Actor Optimizer Settings ===");
if isprop(agent.AgentOptions, "ActorOptimizerOptions")
    disp(agent.AgentOptions.ActorOptimizerOptions);
elseif isprop(agent.AgentOptions, "PolicyOptimizerOptions")
    disp(agent.AgentOptions.PolicyOptimizerOptions);
else
    disp("Actor optimizer settings are not exposed on this agent version.");
end

disp("=== DEBUG: Critic Optimizer Settings ===");
if isprop(agent.AgentOptions, "CriticOptimizerOptions")
    criticOpts = agent.AgentOptions.CriticOptimizerOptions;
    if iscell(criticOpts)
        disp(criticOpts{1});
    else
        disp(criticOpts(1));
    end
else
    disp("Critic optimizer settings are not exposed on this agent version.");
end

printSectionHeader("TRAINING START");
fprintf('Calling train(agent, env, trainOpts) at %s\n', ...
    char(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss")));
trainingStats = train(agent, env, trainOpts);
save(fullfile(saveDir, "trainingStats.mat"), "trainingStats");

% Save training metadata for reproducibility: Config snapshot, seed, and MATLAB/toolbox info
metadata = struct();
metadata.Config = Config; %#ok<NASGU>
metadata.random_seed = Config.random_seed;
metadata.datetime = char(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss"));
try
    metadata.matlab_ver = ver();
catch
    metadata.matlab_ver = [];
end
save(fullfile(saveDir, "training_metadata.mat"), "metadata");

printSectionHeader("TRAINING SUMMARY");
fprintf('trainingStats saved to      : %s\n', fullfile(saveDir, "trainingStats.mat"));
summarizeTrainingStats(trainingStats);
fprintf('trainSAC.m finished at %s\n', char(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss")));

function printSectionHeader(titleText)
fprintf('\n===============================================================\n');
fprintf('%s\n', titleText);
fprintf('===============================================================\n');
end

function text = formatNumericVector(values, formatSpec)
numericValues = double(values(:).');
parts = arrayfun(@(value) sprintf(formatSpec, value), numericValues, "UniformOutput", false);
text = ['[', strjoin(parts, ', '), ']'];
end

function value = readOptionValue(optionsObject, propertyName)
value = NaN;
if isprop(optionsObject, propertyName)
    rawValue = optionsObject.(propertyName);
    if isnumeric(rawValue) && isscalar(rawValue)
        value = double(rawValue);
    end
end
end

function value = readOptimizerValue(optionsObject, optimizerPropertyNames, fieldName)
value = NaN;
propertyNames = string(optimizerPropertyNames);

for nameIdx = 1:numel(propertyNames)
    propertyName = char(propertyNames(nameIdx));
    if ~isprop(optionsObject, propertyName)
        continue;
    end

    optimizerOptions = optionsObject.(propertyName);
    if iscell(optimizerOptions)
        optimizerOptions = optimizerOptions{1};
    elseif numel(optimizerOptions) > 1
        optimizerOptions = optimizerOptions(1);
    end

    if isprop(optimizerOptions, fieldName)
        rawValue = optimizerOptions.(fieldName);
        if isnumeric(rawValue) && isscalar(rawValue)
            value = double(rawValue);
            return;
        end
    end
end
end

function groupCounts = currentGroupCounts(env)
groupMembers = ChannelStateProcessor.groupUsers(env.eMBBPRBCQI);
groupCounts = cellfun(@numel, groupMembers(:)).';
end

function summarizeTrainingStats(trainingStats)
fprintf('trainingStats class         : %s\n', class(trainingStats));

if istable(trainingStats)
    variableNames = string(trainingStats.Properties.VariableNames);
    fprintf('Logged episodes             : %d\n', height(trainingStats));
    fprintf('Available stat columns      : %s\n', strjoin(cellstr(variableNames), ', '));
else
    variableNames = string(fieldnames(trainingStats));
    fprintf('Available stat fields       : %s\n', strjoin(cellstr(variableNames), ', '));
end

summarizeSeries(trainingStats, variableNames, ...
    ["EpisodeReward", "EpisodeCumulativeReward", "Reward"], ...
    'Episode reward summary    ');
summarizeSeries(trainingStats, variableNames, ...
    ["AverageReward", "AverageEpisodeReward"], ...
    'Average reward summary    ');
summarizeSeries(trainingStats, variableNames, ...
    ["EpisodeSteps", "StepsTaken", "EpisodeStepCount"], ...
    'Episode step summary      ');
summarizeSeries(trainingStats, variableNames, ...
    ["EpisodeQ0", "AverageQ0"], ...
    'Q-value summary           ');
end

function summarizeSeries(container, variableNames, candidateNames, label)
series = extractNumericSeries(container, variableNames, candidateNames);
if isempty(series)
    return;
end

fprintf('%s: last=%.4f | mean=%.4f | min=%.4f | max=%.4f\n', ...
    label, series(end), mean(series), min(series), max(series));
end

function series = extractNumericSeries(container, variableNames, candidateNames)
series = [];
matchName = "";
lowerNames = lower(variableNames);

for candidate = string(candidateNames)
    exactIdx = find(lowerNames == lower(candidate), 1);
    if ~isempty(exactIdx)
        matchName = variableNames(exactIdx);
        break;
    end
end

if strlength(matchName) == 0
    for candidate = string(candidateNames)
        fuzzyIdx = find(contains(lowerNames, lower(candidate)), 1);
        if ~isempty(fuzzyIdx)
            matchName = variableNames(fuzzyIdx);
            break;
        end
    end
end

if strlength(matchName) == 0
    return;
end

fieldName = char(matchName);
rawSeries = container.(fieldName);

if iscell(rawSeries)
    try
        rawSeries = cell2mat(rawSeries);
    catch
        return;
    end
end

if ~isnumeric(rawSeries)
    return;
end

series = double(rawSeries(:));
series = series(isfinite(series));
end
