# Diagnostic Task: Inject Deep Debugging and Monitor NaN Crashes

## 🚨 Issue
The SAC agent silently aborts at Episode 6 showing "Training finished" with Q-values stuck at 0. This strongly indicates a silent `NaN` corruption in the network weights or actions, which MATLAB handles by gracefully halting rather than throwing an error.

## 🟢 Phase 1: Inject Heartbeat and NaN Detectors in `RANSlicingEnv.m`
**Goal: Print environment status and catch exactly when actions or rewards become NaN.**

- [ ] **Step 1.1: Add Action/Reward NaN Detectors**
  - Open `RANSlicingEnv.m`.
  - Inside the `step()` method, right after `actionVec = min(1.0, max(0.0, double(action(:))));`, add:
    ```matlab
    % --- DEBUG INJECTION ---
    if any(isnan(actionVec)) || any(isinf(actionVec))
        fprintf('🔥 CRITICAL ERROR: Action contains NaN or Inf at Global Step %d!\n', this.GlobalStepIndex);
        disp(actionVec');
    end
    ```
- [ ] **Step 1.2: Add Heartbeat and Reward Detector**
  - Still inside `step()`, right after `reward = reward / 100.0;` (around the end of the step logic), add:
    ```matlab
    if isnan(reward) || isinf(reward)
        fprintf('🔥 CRITICAL ERROR: Reward is %f at Global Step %d!\n', reward, this.GlobalStepIndex);
    end
    if mod(this.CurrentStep, 200) == 0
        fprintf('[Env Heartbeat] Ep Step: %d | URLLC Q: %.1f | eMBB Q: %.1f | Step Reward: %.4f\n', ...
            this.CurrentStep, sum(this.URLLCGroupQueues), sum(this.eMBBGroupQueues), reward);
    end
    % --- END DEBUG INJECTION ---
    ```

## 🟢 Phase 2: Expose Optimizer Settings and Force Train in `trainSAC.m`
**Goal: Verify if GradientThreshold was actually applied and disable default early stopping.**

- [ ] **Step 2.1: Print Optimizer Settings**
  - Open `trainSAC.m`.
  - Right before `trainingStats = train(agent, env, trainOpts);`, add:
    ```matlab
    disp('=== DEBUG: Actor Optimizer Settings ===');
    disp(agentOpts.ActorOptimizerOptions);
    disp('=== DEBUG: Critic Optimizer Settings ===');
    if iscell(agentOpts.CriticOptimizerOptions)
        disp(agentOpts.CriticOptimizerOptions{1});
    else
        disp(agentOpts.CriticOptimizerOptions(1));
    end
    ```
- [ ] **Step 2.2: Disable Early Stopping**
  - In `trainSAC.m`, modify the `rlTrainingOptions` configuration by adding `"StopTrainingCriteria", "None", ...` so it looks like this:
    ```matlab
    trainOpts = rlTrainingOptions( ...
        "MaxEpisodes", Config.train_max_episodes, ...
        "MaxStepsPerEpisode", Config.Max_Episode_Steps, ...
        "Plots", "training-progress", ...
        "StopTrainingCriteria", "None", ...
        "SaveAgentCriteria", "EpisodeCount", ...
    ```

> 🛑 **Final Check Point:**
> Run `matlab -batch "smokeTest"`. Ensure there are no syntax errors.