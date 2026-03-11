# Iterative Task: Implement 3GPP FTP3 Bursty Traffic Model & Queue-based Reward

## 🎯 Task Overview
The current environment uses a static, randomly generated `eMBB_QoS_req` to evaluate eMBB satisfaction. This does not reflect realistic 5G traffic. 
This task will replace the static QoS with a 3GPP FTP3 bursty traffic model (Poisson arrivals with Pareto-distributed file sizes) for eMBB users. Furthermore, the reward function will be updated to penalize the total eMBB queue backlog, teaching the agent to maintain system stability under bursty loads.

**Development Guidelines**: Execute from Phase 1 to Phase 3. Use `matlab -batch` for Check Points and fix any errors before proceeding. Avoid using functions from Statistics Toolbox (like `random('pareto')`) to ensure compatibility; use the Inverse Transform Sampling method instead.

---

## 🟢 Phase 1: Update `Config.m` for FTP3 Parameters
**Goal: Define FTP3 arrival rates, file size parameters, and queue penalty scale.**

- [x] **Step 1.1: Remove Old QoS Parameters**
  - Open `Config.m`.
  - Delete `eMBB_QoS_min` and `eMBB_QoS_max`.
- [x] **Step 1.2: Add FTP3 Parameters**
  - In the "Traffic and QoS parameters" block, add:
    ```matlab
    lambda_embb (1,1) double = 0.05      % Probability of a new file arriving per mini-slot per group
    embb_xm_bits (1,1) double = 2e6      % Minimum file size (xm) in bits (e.g., 250 KB)
    embb_alpha (1,1) double = 1.2        % Pareto shape parameter (heavy-tailed)
    embb_penalty_scale (1,1) double = 2e-7 % Reward penalty scaling factor for queue length
    ```

> 🛑 **Phase 1 Check Point:**
> Run `matlab -batch "Config.lambda_embb, Config.embb_xm_bits"`. Ensure it outputs 0.05 and 2000000.

---

## 🟢 Phase 2: Overhaul Traffic Generation in `RANSlicingEnv.m`
**Goal: Inject bursty FTP3 files into eMBB queues dynamically during steps.**

- [x] **Step 2.1: Clean up old QoS references**
  - Open `RANSlicingEnv.m`.
  - Remove the property `eMBBQoS_req` from the class definition.
  - In `reset()`, remove any code that initializes `this.eMBBQoS_req`.
  - In `reset()`, explicitly set `this.eMBBGroupQueues = zeros(Config.N_g, 1);`.
- [x] **Step 2.2: Implement FTP3 File Arrivals in `step()`**
  - Inside the `step()` method, right before the channel processing / loop starts, add the FTP3 arrival logic using Inverse Transform Sampling for the Pareto distribution:
    ```matlab
    % 3GPP FTP3 Bursty Traffic Arrivals for eMBB
    for g = 1:Config.N_g
        if rand() < Config.lambda_embb
            u = rand();
            % Inverse transform sampling for Pareto: x = xm / (U^(1/alpha))
            fileSizeBits = Config.embb_xm_bits / (u ^ (1 / Config.embb_alpha));
            this.eMBBGroupQueues(g) = this.eMBBGroupQueues(g) + fileSizeBits;
        end
    end
    ```
- [x] **Step 2.3: Redefine Logged eMBB Satisfaction**
  - In `step()`, find the old `logged.embb_satisfaction` calculation.
  - Replace it with a queue clearance ratio (so downstream scripts like `testAgent.m` still get a 0~1 percentage):
    ```matlab
    totalEmbbQueue = sum(this.eMBBGroupQueues) + sum(embbActualRates) * miniSlotDuration;
    if totalEmbbQueue > 0
        logged.embb_satisfaction = (sum(embbActualRates) * miniSlotDuration) / totalEmbbQueue;
    else
        logged.embb_satisfaction = 1.0; % Perfectly satisfied if queue is empty
    end
    ```

> 🛑 **Phase 2 Check Point:**
> Run `matlab -batch "smokeTest"`. Ensure 140 steps complete without undefined variable errors.

---

## 🟢 Phase 3: Update `calculateReward.m` for Queue-based Penalties
**Goal: Shift the RL objective from QoS satisfaction to queue backlog minimization.**

- [x] **Step 3.1: Modify Reward Logic**
  - Open `calculateReward.m`.
  - Remove the `embbSatisfaction` calculation block (e.g., `sum(embbRates) / sum(QoS)`).
  - Replace it with a queue length penalty calculation:
    ```matlab
    % eMBB Queue Backlog Penalty
    totalEmbbQueueBits = sum(env.eMBBGroupQueues);
    embbQueuePenalty = -Config.embb_penalty_scale * totalEmbbQueueBits;
    ```
- [x] **Step 3.2: Update Final Reward Formula**
  - Change the final `reward` calculation to sum the URLLC penalty, the new eMBB queue penalty, and the fairness metric.
    ```matlab
    reward = urllcReward + embbQueuePenalty + 0.1 * embbFairness;
    ```
    *(Note: Ensure `embbFairness` calculation remains unchanged, keeping its small positive incentive for spatial fairness).*

> 🛑 **Final Check Point (Phase 3):**
> Run `matlab -batch "testAgent"`. The script should run and output the episode logs. The Active eMBB queues should now show massive bursts (heavy-tailed file sizes) arriving dynamically across steps. Mark task as complete if successful.
