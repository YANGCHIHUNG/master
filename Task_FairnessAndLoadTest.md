# Task: Fix Fairness Calculation and Implement Load Stress Test

## 🟢 Phase 1: Fix Jain's Fairness Logic in `calculateReward.m`
**Goal: Only calculate fairness for active eMBB groups (groups with queue > 0). If all queues are empty, fairness should be 1.0 to reward the agent for perfectly clearing the system.**

- [ ] **Step 1.1: Modify Fairness Logic**
  - Open `calculateReward.m`.
  - Locate the fairness calculation block (around line 38):
    ```matlab
    if isempty(rates)
        fFair = 0.0;
    else
        numerator = sum(rates) ^ 2;
        denominator = numel(rates) * sum(rates .^ 2);
    ...
    ```
  - Replace that entire block with the following logic:
    ```matlab
    % Only calculate fairness for groups that actually have data to transmit
    activeIdx = queueBits > 0;
    activeRates = rates(activeIdx);
    
    if isempty(activeRates) || sum(activeRates) == 0
        % If no queues are active, the agent perfectly cleared the system -> Perfect fairness
        fFair = 1.0; 
    else
        numerator = sum(activeRates) ^ 2;
        denominator = numel(activeRates) * sum(activeRates .^ 2);
        if denominator <= eps
            fFair = 0.0;
        else
            fFair = numerator / denominator;
        end
    end
    ```

## 🟢 Phase 2: Create Load Stress Test Script `evaluateLoad.m`
**Goal: Create an automated script to test the agent's generalization across different traffic loads and plot the results.**

- [ ] **Step 2.1: Create `evaluateLoad.m`**
  - Create a new file named `evaluateLoad.m` in the root directory.
  - Insert the following complete script:

```matlab
% evaluateLoad.m
% Evaluate the trained SAC agent under different eMBB traffic loads.

clear all; close all; clc;

disp('===============================================================');
disp('Starting Automated Load Stress Test (Generalization Evaluation)');
disp('===============================================================');

% 1. Find the latest saved agent
modelDir = Config.train_save_dir;
matFiles = dir(fullfile(modelDir, 'Agent*.mat'));
if isempty(matFiles)
    error('No trained Agent*.mat found in %s.', modelDir);
end
[~, idx] = sort([matFiles.datenum], 'descend');
latestModelPath = fullfile(modelDir, matFiles(idx(1)).name);
fprintf('Loading latest model: %s\n', matFiles(idx(1)).name);
load(latestModelPath, 'agent');

% 2. Define Test Loads (lambda_embb)
testLoads = [0.001, 0.003, 0.005, 0.010]; 
loadNames = {'Light (0.001)', 'Medium (0.003)', 'Heavy (0.005)', 'Extreme (0.010)'};

% Metrics storage
urllc_fail_rates = zeros(1, length(testLoads));
embb_satisfaction = zeros(1, length(testLoads));
embb_fairness = zeros(1, length(testLoads));

numSteps = Config.Max_Episode_Steps;

for i = 1:length(testLoads)
    currentLoad = testLoads(i);
    fprintf('\nEvaluating Load %d/%d: lambda = %.3f %s...\n', i, length(testLoads), currentLoad, loadNames{i});
    
    % Temporarily override Config (We use dynamic assignment in Env if possible, 
    % but standard Config.m requires a workaround or Env recreation. 
    % We will modify the default property inside the environment instance directly.)
    
    env = RANSlicingEnv();
    env.lambda_embb = currentLoad; % Override the load
    
    obs = reset(env);
    
    totalURLLCFailures = 0;
    totalEMBBSat = 0;
    totalFairness = 0;
    validFairnessSteps = 0;
    
    for step = 1:numSteps
        action = getAction(agent, {obs});
        actionVec = action{1};
        [nextObs, reward, isDone, loggedSignals] = env.step(actionVec);
        obs = nextObs;
        
        % Collect Metrics
        if loggedSignals.URLLC_Delay > Config.urllc_req_delay
            totalURLLCFailures = totalURLLCFailures + 1;
        end
        totalEMBBSat = totalEMBBSat + loggedSignals.eMBB_Satisfaction;
        
        if ~isnan(loggedSignals.eMBB_Fairness) && loggedSignals.eMBB_Fairness > 0
            totalFairness = totalFairness + loggedSignals.eMBB_Fairness;
            validFairnessSteps = validFairnessSteps + 1;
        end
    end
    
    % Calculate averages
    urllc_fail_rates(i) = (totalURLLCFailures / numSteps) * 100;
    embb_satisfaction(i) = totalEMBBSat / numSteps;
    if validFairnessSteps > 0
        embb_fairness(i) = totalFairness / validFairnessSteps;
    else
        embb_fairness(i) = 1.0;
    end
    
    fprintf('  -> URLLC Fail Rate : %.2f %%\n', urllc_fail_rates(i));
    fprintf('  -> Average eMBB Sat: %.2f %%\n', embb_satisfaction(i));
    fprintf('  -> Average Fairness: %.4f\n', embb_fairness(i));
end

% 3. Plot Results
figure('Name', 'Model Generalization under Traffic Load', 'Position', [100, 100, 1200, 400]);

subplot(1,3,1);
plot(testLoads, urllc_fail_rates, '-or', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
grid on; xlabel('eMBB Arrival Rate (\lambda)'); ylabel('URLLC Failure Rate (%)');
title('URLLC Reliability vs. Load');
ylim([0, max(10, max(urllc_fail_rates)*1.2)]);

subplot(1,3,2);
plot(testLoads, embb_satisfaction, '-ob', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
grid on; xlabel('eMBB Arrival Rate (\lambda)'); ylabel('Average Satisfaction (%)');
title('eMBB Satisfaction vs. Load');
ylim([0, 100]);

subplot(1,3,3);
plot(testLoads, embb_fairness, '-og', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'g');
grid on; xlabel('eMBB Arrival Rate (\lambda)'); ylabel('Jain''s Fairness Index');
title('eMBB Fairness vs. Load');
ylim([0, 1.1]);

disp('Evaluation complete. Please check the generated plots.');