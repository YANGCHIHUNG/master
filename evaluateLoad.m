%EVALUATELOAD Evaluate the trained SAC agent under different eMBB traffic loads.

clear;
close all;
clc;

fprintf('===============================================================\n');
fprintf('Starting Automated Load Stress Test (Generalization Evaluation)\n');
fprintf('===============================================================\n');

rng(Config.random_seed);
verifyFairnessFix();

[agent, latestModelPath] = loadLatestAgent(fullfile(pwd, char(Config.train_save_dir)));

testLoads = [0.001, 0.003, 0.005, 0.010];
loadNames = {'Light (0.001)', 'Medium (0.003)', 'Heavy (0.005)', 'Extreme (0.010)'};
numEpisodes = 5;

urllcFailRates = zeros(1, numel(testLoads));
embbSatisfaction = zeros(1, numel(testLoads));
embbFairness = zeros(1, numel(testLoads));

fprintf('Loaded latest model         : %s\n', latestModelPath);
fprintf('Episodes per load           : %d\n', numEpisodes);
fprintf('Steps per episode           : %d\n', Config.Max_Episode_Steps);
fprintf('===============================================================\n');

for loadIdx = 1:numel(testLoads)
    currentLoad = testLoads(loadIdx);
    fprintf('\nEvaluating load %d/%d: lambda = %.3f %s\n', ...
        loadIdx, numel(testLoads), currentLoad, loadNames{loadIdx});

    env = RANSlicingEnv();
    env.lambda_embb = currentLoad;

    totalURLLCFailures = 0;
    totalEMBBSat = 0.0;
    totalFairness = 0.0;
    totalSteps = 0;

    for episodeIdx = 1:numEpisodes
        obs = reset(env);

        for stepIdx = 1:Config.Max_Episode_Steps
            actionCell = getAction(agent, obs);
            action = actionCell{1};
            [obs, ~, isDone, loggedSignals] = step(env, action);

            totalURLLCFailures = totalURLLCFailures + double(loggedSignals.is_urllc_failed);
            totalEMBBSat = totalEMBBSat + 100.0 * loggedSignals.embb_satisfaction;
            totalFairness = totalFairness + loggedSignals.embb_fairness;
            totalSteps = totalSteps + 1;

            if isDone
                break;
            end
        end

        fprintf('  Completed episode %d / %d\n', episodeIdx, numEpisodes);
    end

    urllcFailRates(loadIdx) = 100.0 * totalURLLCFailures / max(totalSteps, 1);
    embbSatisfaction(loadIdx) = totalEMBBSat / max(totalSteps, 1);
    embbFairness(loadIdx) = totalFairness / max(totalSteps, 1);

    fprintf('  -> URLLC Fail Rate : %.2f %%\n', urllcFailRates(loadIdx));
    fprintf('  -> Average eMBB Sat: %.2f %%\n', embbSatisfaction(loadIdx));
    fprintf('  -> Average Fairness: %.4f\n', embbFairness(loadIdx));
end

assert(all(isfinite(urllcFailRates)), 'evaluateLoad:InvalidFailureRate', ...
    'Found NaN/Inf in URLLC failure rates.');
assert(all(isfinite(embbSatisfaction)), 'evaluateLoad:InvalidSatisfaction', ...
    'Found NaN/Inf in eMBB satisfaction values.');
assert(all(isfinite(embbFairness)), 'evaluateLoad:InvalidFairness', ...
    'Found NaN/Inf in fairness values.');
assert(all(urllcFailRates >= 0 & urllcFailRates <= 100), ...
    'evaluateLoad:FailureRateOutOfRange', ...
    'URLLC failure rates must stay in [0, 100].');
assert(all(embbSatisfaction >= 0 & embbSatisfaction <= 100), ...
    'evaluateLoad:SatisfactionOutOfRange', ...
    'eMBB satisfaction must stay in [0, 100].');
assert(all(embbFairness >= 0 & embbFairness <= 1 + 1e-9), ...
    'evaluateLoad:FairnessOutOfRange', ...
    'Fairness must stay in [0, 1].');

fig = figure( ...
    'Name', 'Model Generalization under Traffic Load', ...
    'Position', [100, 100, 1200, 400], ...
    'Visible', 'off', ...
    'Color', 'w');

subplot(1, 3, 1);
plot(testLoads, urllcFailRates, '-or', 'LineWidth', 2, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'r');
grid on;
xlabel('eMBB Arrival Rate (\lambda)');
ylabel('URLLC Failure Rate (%)');
title('URLLC Reliability vs. Load');
ylim([0, max(10, max(urllcFailRates) * 1.2)]);

subplot(1, 3, 2);
plot(testLoads, embbSatisfaction, '-ob', 'LineWidth', 2, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'b');
grid on;
xlabel('eMBB Arrival Rate (\lambda)');
ylabel('Average Satisfaction (%)');
title('eMBB Satisfaction vs. Load');
ylim([0, 100]);

subplot(1, 3, 3);
plot(testLoads, embbFairness, '-og', 'LineWidth', 2, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'g');
grid on;
xlabel('eMBB Arrival Rate (\lambda)');
ylabel('Jain''s Fairness Index');
title('eMBB Fairness vs. Load');
ylim([0, 1.1]);

saveas(fig, 'Load_Evaluation_Report.png');
close(fig);

save('Load_Evaluation_Results.mat', ...
    'testLoads', 'loadNames', 'numEpisodes', ...
    'urllcFailRates', 'embbSatisfaction', 'embbFairness', ...
    'latestModelPath');

fprintf('\n===============================================================\n');
fprintf('Load stress test completed successfully.\n');
fprintf('Saved plot                  : %s\n', fullfile(pwd, 'Load_Evaluation_Report.png'));
fprintf('Saved metrics               : %s\n', fullfile(pwd, 'Load_Evaluation_Results.mat'));
fprintf('===============================================================\n');

function verifyFairnessFix()
perfectReward = calculateReward(0.0, Config.tau_req, zeros(Config.N_g, 1), zeros(Config.N_g, 1));
expectedPerfectReward = 0.1 / 100.0;
assert(abs(perfectReward - expectedPerfectReward) <= 1e-12, ...
    'evaluateLoad:FairnessFixMissing', ...
    'Reward fairness sanity check failed for perfectly cleared queues.');

testRates = [100.0; 1.0; 2.0];
testQueues = [0.0; 10.0; 10.0];
activeFairness = (sum([1.0; 2.0]) ^ 2) / (2 * sum([1.0; 2.0] .^ 2));
queuePenalty = -Config.embb_penalty_scale * sum(testQueues);
expectedReward = (queuePenalty + 0.1 * activeFairness) / 100.0;
actualReward = calculateReward(0.0, Config.tau_req, testRates, testQueues);
assert(abs(actualReward - expectedReward) <= 1e-12, ...
    'evaluateLoad:ActiveFairnessMismatch', ...
    'Reward fairness sanity check failed for active eMBB groups.');
end

function [agent, latestAgentPath] = loadLatestAgent(saveDir)
if ~exist(saveDir, 'dir')
    error('evaluateLoad:MissingModelDirectory', ...
        'Model directory does not exist: %s', saveDir);
end

agentFiles = dir(fullfile(saveDir, 'Agent*.mat'));
if isempty(agentFiles)
    error('evaluateLoad:NoAgentCheckpoint', ...
        'No Agent*.mat files were found in %s.', saveDir);
end

[~, sortIdx] = sort([agentFiles.datenum], 'descend');
latestFile = agentFiles(sortIdx(1));
latestAgentPath = fullfile(latestFile.folder, latestFile.name);

loadedData = load(latestAgentPath);
if ~isfield(loadedData, 'saved_agent')
    error('evaluateLoad:MissingSavedAgent', ...
        'Checkpoint %s does not contain a saved_agent variable.', latestAgentPath);
end

agent = loadedData.saved_agent;
end
