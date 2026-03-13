# Task: Synchronize Spec, Tests, and Fix RL State/Trace Integrity

## 🟢 Phase 1: Fix RL Observation Blindness (Logarithmic Normalization)
**Goal: Prevent state saturation from Pareto heavy-tail traffic using Log10 normalization.**

- [ ] **Step 1.1: Modify `RANSlicingEnv.m` Observation Logic**
  - Open `RANSlicingEnv.m`. Locate the `getObservation` method.
  - Change the linear normalization to logarithmic. Replace the URLLC and eMBB normalization lines with:
    ```matlab
    % Use Log10 normalization to handle heavy-tailed queue sizes without saturation blinding
    % Assuming max reasonable queue is around 10^9 bits (log10(10^9) = 9)
  % Prefer using Config constants to avoid magic numbers:
  normalizedURLLC = log10(1 + this.URLLCGroupQueues) / Config.obs_urlllc_log_scale; % Scale down max expected queue
  normalizedEMBB = log10(1 + this.eMBBGroupQueues) / Config.obs_embb_log_scale;  % Scale down max expected queue
    
    % Ensure strict [0, 1] bounds just in case of astronomical surges
    normalizedURLLC = min(1.0, max(0.0, normalizedURLLC));
    normalizedEMBB = min(1.0, max(0.0, normalizedEMBB));
    ```

## 🟢 Phase 2: Fix Episode Independence (Channel Trace Randomization)
**Goal: Ensure IID assumption by randomizing channel trace start position on every reset.**

- [ ] **Step 2.1: Modify `reset` in `RANSlicingEnv.m`**
  - Open `RANSlicingEnv.m`. Locate the `reset` method.
  - Before resetting queues, add logic to pick a random starting index within safe bounds:
    ```matlab
    % Randomize starting point to ensure episode independence (IID)
    maxStartIndex = length(this.ChannelTraces.URLLC) - Config.Max_Episode_Steps * 7 - 1;
    if maxStartIndex > 1
        this.GlobalStepIndex = randi([1, maxStartIndex]);
    else
        this.GlobalStepIndex = 1;
    end
    ```

## 🟢 Phase 3: Sync Unit Tests with Current Reward Logic
**Goal: Update `Tests.m` to pass with the new scaled, penalty-based reward structure.**

- [ ] **Step 3.1: Update `Tests.m`**
  - Open `Tests.m`.
  - Locate all `testReward...` functions (e.g., `testRewardDelayViolation`, `testRewardSatisfaction`).
  - Update the expected logic in the assertions to match `calculateReward.m` (it uses `-embbQueueBits * Config.embb_penalty_scale`, adds `Config.reward_fairness_weight * fFair` if no delay violation, and divides the final result by `Config.reward_scale`).
  - Specifically, for empty queues and no violations, expected reward is `(0 + Config.reward_fairness_weight * 1.0) / Config.reward_scale`.

> 🛑 **Final Check Point:**
> Run `matlab -batch "results = runtests('Tests.m'); assert(all([results.Passed]));"`. Ensure all tests pass.