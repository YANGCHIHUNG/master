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

rng(Config.random_seed);
env = RANSlicingEnv();

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
agent = rlSACAgent(obsInfo, actInfo);

agentOpts = agent.AgentOptions;
if isprop(agentOpts, "MiniBatchSize")
    agentOpts.MiniBatchSize = Config.sac_minibatch_size;
end

if isprop(agentOpts, "ActorOptimizerOptions")
    if isprop(agentOpts.ActorOptimizerOptions, "LearnRate")
        agentOpts.ActorOptimizerOptions.LearnRate = Config.sac_actor_learn_rate;
    end
elseif isprop(agentOpts, "PolicyOptimizerOptions")
    if isprop(agentOpts.PolicyOptimizerOptions, "LearnRate")
        agentOpts.PolicyOptimizerOptions.LearnRate = Config.sac_actor_learn_rate;
    end
end

if isprop(agentOpts, "CriticOptimizerOptions")
    criticOpts = agentOpts.CriticOptimizerOptions;
    if iscell(criticOpts)
        for idx = 1:numel(criticOpts)
            if isprop(criticOpts{idx}, "LearnRate")
                criticOpts{idx}.LearnRate = Config.sac_critic_learn_rate;
            end
        end
        agentOpts.CriticOptimizerOptions = criticOpts;
    else
        if isprop(criticOpts, "LearnRate")
            criticOpts.LearnRate = Config.sac_critic_learn_rate;
            agentOpts.CriticOptimizerOptions = criticOpts;
        end
    end
end

agent.AgentOptions = agentOpts;

saveDir = fullfile(pwd, char(Config.train_save_dir));
if ~exist(saveDir, "dir")
    mkdir(saveDir);
end

trainOpts = rlTrainingOptions( ...
    "MaxEpisodes", Config.train_max_episodes, ...
    "MaxStepsPerEpisode", Config.M_mini_slots, ...
    "Plots", "training-progress", ...
    "SaveAgentCriteria", "EpisodeCount", ...
    "SaveAgentValue", Config.train_save_every_episodes, ...
    "SaveAgentDirectory", saveDir);

trainingStats = train(agent, env, trainOpts);
save(fullfile(saveDir, "trainingStats.mat"), "trainingStats");
