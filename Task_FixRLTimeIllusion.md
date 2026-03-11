# Diagnostic and Repair Task: Fix RL Time Illusion and Queue Reset Flaw

## 🚨 Observed Phenomenon
During the `trainSAC` execution, the Episode Q0 curve stays strictly at 0 for hundreds of episodes, and the agent fails to learn how to drain the bursty FTP3 queues. 

## 🔍 Hypothesized Root Causes
1. **RL Time Illusion (Warm-up Period)**: `rlSACAgent` has a default `NumWarmStartSteps` (usually 1000). Since our current episode length is only 7 steps (`Config.M_mini_slots`), the agent spends the first ~142 episodes just collecting random data without updating the Q-networks.
2. **Lyapunov Optimization Destroyer (Premature Reset)**: The environment triggers `isDone = true` and calls `reset()` every 7 steps. This magically wipes out the massive eMBB FTP3 queues (e.g., 15M bits) before the agent can suffer the long-term penalty of not draining them. The agent never learns long-term queue stability.

## 🎯 Task Overview
Inspect the project to verify these hypotheses. Then, extend the episode length to 700 steps (100 full slots) while keeping the 1~7 `MiniSlotIndex` cycle intact.

**Development Guidelines**: Execute from Phase 1 to Phase 3. Use `matlab -batch` for Check Points and fix any errors before proceeding.

---

## 🟢 Phase 1: Add Episode Length Parameter in `Config.m`
**Goal: Define a long episode length to allow proper RL training and queue accumulation.**

- [ ] **Step 1.1: Add Parameter**
  - Open `Config.m`.
  - In the "Simulation parameters" block, add a new property:
    `Max_Episode_Steps (1,1) double = 700 % 100 slots (700 mini-slots) per episode`

## 🟢 Phase 2: Fix Continuous Tracking in `RANSlicingEnv.m`
**Goal: Track continuous steps for termination but keep the cyclic 1~7 mini-slot index for physical logic.**

- [ ] **Step 2.1: Add `CurrentStep` State**
  - Open `RANSlicingEnv.m`.
  - In the `properties (Access = private)` block, add: `CurrentStep`
- [ ] **Step 2.2: Update `reset()`**
  - Inside `reset()`, initialize `this.CurrentStep = 0;`.
  - Keep `this.MiniSlotIndex = 1;`.
- [ ] **Step 2.3: Update `step()` logic**
  - At the very beginning of the `step()` method, add `this.CurrentStep = this.CurrentStep + 1;`.
  - At the end of the `step()` method (right before returning outputs), update the `isDone` flag and advance the cyclic mini-slot index:
    ```matlab
    isDone = this.CurrentStep >= Config.Max_Episode_Steps;
    this.MiniSlotIndex = mod(this.CurrentStep, Config.M_mini_slots) + 1;
    ```
  - *Crucial*: Search through `step()` and remove any old logic that sets `isDone = true` or `isDone = (this.MiniSlotIndex >= Config.M_mini_slots)`.

## 🟢 Phase 3: Update Diagnostics & Scripts
**Goal: Make all evaluation scripts aware of the new extended episode length.**

- [ ] **Step 3.1: Update `testAgent.m`**
  - Open `testAgent.m`.
  - Replace all instances of `Config.M_mini_slots` with `Config.Max_Episode_Steps` for the main `for k = 1:...` loop.
  - Fix the summary calculations: e.g., `avgSatisfaction = (sumSatisfaction / Config.Max_Episode_Steps) * 100;` and `Failure Rate` calculations.
- [ ] **Step 3.2: Update `evaluateModel.m`**
  - Open `evaluateModel.m`.
  - Replace `Config.M_mini_slots` with `Config.Max_Episode_Steps` when calculating `totalSteps = numEpisodes * ...;` and in the inner `for k = 1:...` loop.
  - Ensure the output text uses `Config.Max_Episode_Steps` when printing "Steps per episode".

> 🛑 **Final Check Point:**
> Run `matlab -batch "testAgent"`. The script should now run for 700 steps instead of 7, and successfully print the episode summary metrics. Mark task as complete if successful.