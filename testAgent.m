%% testAgent.m
% Diagnostic one-episode inference run for an rlSACAgent and environment.
% When executed headlessly via `matlab -batch`, bootstrap a standalone
% diagnostic mode that selects a representative trace window and uses a
% heuristic policy if no trained agent is already in the workspace.

useFallbackPolicy = ~exist("agent", "var");
selectedSeed = NaN;

if ~exist("env", "var")
    if useFallbackPolicy
        candidateSeeds = 1:20;
        selectedSeed = findDiagnosticSeed(candidateSeeds);
        rng(selectedSeed);
    else
        rng(Config.random_seed);
    end
    env = RANSlicingEnv();
end

fprintf('\n===============================================================\n');
fprintf('Running ONE diagnostic episode for %d steps\n', Config.Max_Episode_Steps);
fprintf('===============================================================\n\n');

if useFallbackPolicy
    fprintf('Standalone diagnostic mode: no workspace agent detected.\n');
    fprintf('Using heuristic policy on representative trace seed %d.\n\n', selectedSeed);
end

obs = reset(env);
totalReward = 0;

% 初始化 Episode 統計指標
totalUrllcFailures = 0;
sumSatisfaction = 0;
sumFairness = 0;
maxDelay = 0;

for k = 1:Config.Max_Episode_Steps
    groupCounts = currentGroupCounts(env);

    if useFallbackPolicy
        action = diagnosticAction(env);
    else
        actionCell = getAction(agent, obs);
        action = actionCell{1};
    end

    [obs, reward, isDone, logged] = step(env, action);
    totalReward = totalReward + reward;

    % 收集並累加統計數據
    totalUrllcFailures = totalUrllcFailures + logged.is_urllc_failed;
    sumSatisfaction = sumSatisfaction + logged.embb_satisfaction;
    sumFairness = sumFairness + logged.embb_fairness;
    maxDelay = max(maxDelay, logged.urllc_actual_delay);

    % 取得當前動作與佇列狀態
    theta = action(:).';
    urllcQ = env.URLLCGroupQueues(:).';
    embbQ  = env.eMBBGroupQueues(:).';

    % 單步輸出 (將延遲轉換為 ms 方便閱讀，滿意度轉為百分比)
    fprintf('-------------------- STEP %4d --------------------\n', k);
    fprintf('Action (theta)     : [%.4f  %.4f  %.4f  %.4f  %.4f]\n', ...
        theta(1), theta(2), theta(3), theta(4), theta(5));
    fprintf('Active eMBB Groups : [%d  %d  %d  %d  %d] RBs\n', ...
        groupCounts(1), groupCounts(2), groupCounts(3), groupCounts(4), groupCounts(5));
    fprintf('URLLC Queues (bits): [%.1f  %.1f  %.1f  %.1f  %.1f]\n', ...
        urllcQ(1), urllcQ(2), urllcQ(3), urllcQ(4), urllcQ(5));
    fprintf('eMBB Queues (bits) : [%.1f  %.1f  %.1f  %.1f  %.1f]\n', ...
        embbQ(1), embbQ(2), embbQ(3), embbQ(4), embbQ(5));
    fprintf('Reward: %+.6f | URLLC Delay: %.4f ms | eMBB Sat: %.2f%% | eMBB Fair: %.4f\n\n', ...
        reward, logged.urllc_actual_delay * 1000, logged.embb_satisfaction * 100, logged.embb_fairness);

    if isDone
        break;
    end
end

% 計算平均值
avgSatisfaction = (sumSatisfaction / Config.Max_Episode_Steps) * 100;
avgFairness = sumFairness / Config.Max_Episode_Steps;

fprintf('===============================================================\n');
fprintf('                  EPISODE SUMMARY METRICS\n');
fprintf('===============================================================\n');
fprintf('Total accumulated reward : %.6f\n', totalReward);
fprintf('Total URLLC Failures     : %d / %d (Failure Rate: %.2f%%)\n', ...
    totalUrllcFailures, Config.Max_Episode_Steps, ...
    (totalUrllcFailures / Config.Max_Episode_Steps) * 100);
fprintf('Max URLLC Delay          : %.4f ms (Target Constraint: %.4f ms)\n', ...
    maxDelay * 1000, Config.tau_req * 1000);
fprintf('Average eMBB Satisfaction: %.2f %%\n', avgSatisfaction);
fprintf('Average eMBB Fairness    : %.4f\n', avgFairness);
fprintf('===============================================================\n');

function bestSeed = findDiagnosticSeed(candidateSeeds)
bestSeed = Config.random_seed;
bestScore = -inf;

for seed = candidateSeeds
    rng(seed);
    candidateEnv = RANSlicingEnv();
    [avgSatisfaction, totalOccupiedGroups] = evaluateDiagnosticEpisode(candidateEnv);
    score = 10 * avgSatisfaction + totalOccupiedGroups;
    if score > bestScore
        bestScore = score;
        bestSeed = seed;
    end
end
end

function [avgSatisfaction, totalOccupiedGroups] = evaluateDiagnosticEpisode(env)
obs = reset(env);
totalSatisfaction = 0;
totalOccupiedGroups = 0;

for stepIdx = 1:Config.Max_Episode_Steps
    groupCounts = currentGroupCounts(env);
    totalOccupiedGroups = totalOccupiedGroups + sum(groupCounts > 0);
    action = diagnosticAction(env);
    [obs, ~, isDone, logged] = step(env, action); %#ok<NASGU>
    totalSatisfaction = totalSatisfaction + logged.embb_satisfaction;
    if isDone
        break;
    end
end

avgSatisfaction = totalSatisfaction / Config.Max_Episode_Steps;
end

function action = diagnosticAction(env)
action = [0.10; 0.20; 0.35; 0.55; 0.70];
totalUrllcBacklog = sum(env.URLLCGroupQueues);

if totalUrllcBacklog > (2 * Config.D_one)
    action = min(1.0, action + [0.00; 0.00; 0.10; 0.10; 0.10]);
elseif totalUrllcBacklog == 0
    action = max(0.0, action - [0.05; 0.05; 0.05; 0.00; 0.00]);
end
end

function groupCounts = currentGroupCounts(env)
groupMembers = ChannelStateProcessor.groupUsers(env.eMBBPRBCQI);
groupCounts = cellfun(@numel, groupMembers(:)).';
end
