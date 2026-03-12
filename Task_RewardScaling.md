# Task: Implement Reward Scaling to Prevent Gradient Explosion

## 🚨 Issue
The SAC agent suffers from gradient explosion at Episode 6. While the eMBB penalty was safely clamped, the `urllcReward` penalty can reach approximately -200 per step due to long queue delays in the 700-step episode. A combined step reward of -250 causes the Q-value targets to reach -25,000, creating huge MSE loss and `NaN` gradients.

## 🟢 Phase 1: Scale the Reward
**Goal: Compress the final reward into a stable range [-2.5, 1] for SAC networks.**

- [ ] **Step 1.1: Modify `calculateReward.m`**
  - Open `calculateReward.m`.
  - At the very end of the script, right before the `end` keyword of the function, add the following line to scale the composite reward:
    ```matlab
    % Scale reward to prevent gradient explosion in SAC neural networks
    reward = reward / 100.0;
    ```

> 🛑 **Final Check Point:**
> Run `matlab -batch "smokeTest"`. Ensure there are no syntax errors and mark the task as complete.