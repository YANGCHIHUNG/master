# Task: Add Gradient Clipping and Reduce Update Frequency in SAC

## 🚨 Issue
The environment reward scaling is working perfectly (episode reward ~ -300), but the SAC agent still diverges and triggers "Training finished" at Episode 6. This is caused by over-updating (700 updates per episode) without gradient clipping, eventually leading to `NaN` weights.

## 🟢 Phase 1: Modify `trainSAC.m`
**Goal: Limit update frequency to 10 steps and enforce a GradientThreshold of 1.0.**

- [ ] **Step 1.1: Reduce Update Frequency**
  - Open `trainSAC.m`.
  - In the `agentOpts` configuration section, add the following line:
    ```matlab
    agentOpts.NumStepsPerUpdate = 10;
    ```
- [ ] **Step 1.2: Add Gradient Clipping to Actor**
  - Locate the `ActorOptimizerOptions` (or `PolicyOptimizerOptions`) block. Add:
    ```matlab
    agentOpts.ActorOptimizerOptions.GradientThreshold = 1.0;
    ```
    *(Note: if using `PolicyOptimizerOptions`, add it there instead).*
- [ ] **Step 1.3: Add Gradient Clipping to Critic**
  - Locate the `CriticOptimizerOptions` loop block. Add `GradientThreshold = 1.0;` to both the cell array and normal array branches:
    ```matlab
        % Inside the if iscell(criticOpts) loop:
        criticOpts{idx}.GradientThreshold = 1.0;
        
        % Inside the else loop:
        criticOpts(idx).GradientThreshold = 1.0;
    ```

> 🛑 **Final Check Point:**
> Run `matlab -batch "smokeTest"`. Ensure the script compiles and runs without errors.