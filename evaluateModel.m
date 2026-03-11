%% evaluateModel.m
% Monte Carlo evaluation for the latest trained SAC agent.

clear;
clc;

rng(Config.random_seed);

numEpisodes = 500;
env = RANSlicingEnv();

[agent, latestAgentPath] = loadLatestAgent(fullfile(pwd, char(Config.train_save_dir)));

totalSteps = numEpisodes * Config.Max_Episode_Steps;
allUrllcDelays = zeros(totalSteps, 1);
allEmbbSatisfactions = zeros(totalSteps, 1);
allUrllcFailures = 0;
globalStep = 1;

fprintf('\n===============================================================\n');
fprintf('Monte Carlo Evaluation Started\n');
fprintf('===============================================================\n');
fprintf('Episodes            : %d\n', numEpisodes);
fprintf('Steps per episode   : %d\n', Config.Max_Episode_Steps);
fprintf('Loaded agent        : %s\n', latestAgentPath);
fprintf('===============================================================\n\n');

for ep = 1:numEpisodes
    obs = reset(env);

    for k = 1:Config.Max_Episode_Steps
        actionCell = getAction(agent, obs);
        action = actionCell{1};
        [obs, ~, isDone, logged] = step(env, action);

        allUrllcDelays(globalStep) = logged.urllc_actual_delay * 1000;
        allEmbbSatisfactions(globalStep) = logged.embb_satisfaction * 100;
        allUrllcFailures = allUrllcFailures + double(logged.is_urllc_failed);
        globalStep = globalStep + 1;

        if isDone
            break;
        end
    end

    if mod(ep, 50) == 0 || ep == numEpisodes
        fprintf('Completed episode %d / %d\n', ep, numEpisodes);
    end
end

validSteps = globalStep - 1;
if validSteps <= 0
    error('evaluateModel:NoDataCollected', 'No evaluation samples were collected.');
end

allUrllcDelays = allUrllcDelays(1:validSteps);
allEmbbSatisfactions = allEmbbSatisfactions(1:validSteps);

overallFailureRate = (allUrllcFailures / validSteps) * 100;
urllcDelayP99 = computePercentile(allUrllcDelays, 99);
averageEmbbSatisfaction = mean(allEmbbSatisfactions);

fprintf('\n==================== Statistical Summary ====================\n');
fprintf('Total Episodes tested      : %d\n', numEpisodes);
fprintf('Total Steps recorded       : %d\n', validSteps);
fprintf('Overall URLLC Failure Rate : %.4f %%\n', overallFailureRate);
fprintf('99th Percentile URLLC Delay: %.6f ms\n', urllcDelayP99);
fprintf('Average eMBB Satisfaction  : %.4f %%\n', averageEmbbSatisfaction);
fprintf('=============================================================\n');

fig = figure( ...
    'Name', 'Evaluation Results', ...
    'Position', [100, 100, 1200, 400], ...
    'Visible', 'off', ...
    'Color', 'w');

subplot(1, 2, 1);
[cdfX, cdfY] = empiricalCdf(allUrllcDelays);
plot(cdfX, cdfY, 'b-', 'LineWidth', 1.8);
hold on;
plot([Config.tau_req, Config.tau_req] * 1000, [0, 1], 'r--', 'LineWidth', 1.5);
hold off;
grid on;
title('CDF of URLLC Delay');
xlabel('Delay (ms)');
ylabel('Probability');
legend('Empirical CDF', 'Target Constraint', 'Location', 'southeast');

subplot(1, 2, 2);
histogram(allEmbbSatisfactions, 'Normalization', 'probability', 'FaceColor', [0.2, 0.5, 0.8]);
grid on;
title('Distribution of eMBB Satisfaction');
xlabel('Satisfaction (%)');
ylabel('Probability');

saveas(gcf, 'Evaluation_Report.png');
close(fig);

function [agent, latestAgentPath] = loadLatestAgent(saveDir)
if ~exist(saveDir, 'dir')
    error('evaluateModel:MissingModelDirectory', ...
        'Model directory does not exist: %s', saveDir);
end

agentFiles = dir(fullfile(saveDir, 'Agent*.mat'));
if isempty(agentFiles)
    error('evaluateModel:NoAgentCheckpoint', ...
        'No Agent*.mat files were found in %s.', saveDir);
end

[~, nameOrder] = sort({agentFiles.name});
agentFiles = agentFiles(nameOrder);
[~, timeOrder] = sort([agentFiles.datenum]);
latestFile = agentFiles(timeOrder(end));

if isfield(latestFile, 'folder') && strlength(string(latestFile.folder)) > 0
    latestAgentPath = fullfile(latestFile.folder, latestFile.name);
else
    latestAgentPath = fullfile(saveDir, latestFile.name);
end

loadedData = load(latestAgentPath);
if ~isfield(loadedData, 'saved_agent')
    error('evaluateModel:MissingSavedAgent', ...
        'Checkpoint %s does not contain a saved_agent variable.', latestAgentPath);
end

agent = loadedData.saved_agent;
end

function percentileValue = computePercentile(values, percentile)
if exist('prctile', 'file') == 2
    percentileValue = prctile(values, percentile);
    return;
end

sortedValues = sort(values(:));
sampleCount = numel(sortedValues);
if sampleCount == 0
    percentileValue = NaN;
    return;
end

rank = 1 + (sampleCount - 1) * (percentile / 100);
lowerIndex = floor(rank);
upperIndex = ceil(rank);

if lowerIndex == upperIndex
    percentileValue = sortedValues(lowerIndex);
else
    weight = rank - lowerIndex;
    percentileValue = sortedValues(lowerIndex) + ...
        weight * (sortedValues(upperIndex) - sortedValues(lowerIndex));
end
end

function [xValues, yValues] = empiricalCdf(samples)
xValues = sort(samples(:));
sampleCount = numel(xValues);
yValues = (1:sampleCount)' / sampleCount;
end
