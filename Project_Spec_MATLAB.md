# 📄 Project Specification and Prompt Document (`Project_Spec_MATLAB.md`)

## Part 1: Project Overview & Global Context

**【Project Goal】**
Build a MATLAB-based 5G Radio Access Network (RAN) resource scheduling simulator, utilizing the Reinforcement Learning Toolbox's Soft Actor-Critic (SAC) algorithm to optimize resource slicing for eMBB and URLLC traffic under frequency-selective fading environments.

**【Tech Stack】**

* MATLAB
* Reinforcement Learning Toolbox (Custom RL Environment & SAC Agent)
* 5G Toolbox (Physical Layer / Channel Modeling)
* Deep Learning Toolbox (Underlying Neural Networks)

**【Coding Style Requirements】**

* Comprehensively use MATLAB Class definition (`classdef`) with strict property validation (e.g., `properties (Constant)`).
* All Classes and Methods must include standard MATLAB Help Text (comments immediately following the function definition) clearly explaining Input/Output dimensions and their physical significance.

**【Directory Structure】**

Please strictly follow this structure when generating and organizing code. All modules MUST be in the same working directory or path to utilize `Config.m`:

* `Config.m` (Centralized configuration class for ALL physical and RL hyperparameters)
* `RANSlicingEnv.m` (Main environment class inheriting `rl.env.MATLABEnvironment`)
* `ChannelStateProcessor.m` (Physical layer: Channel gain, SNR calculation, MCS mapping module)
* `calculateReward.m` (Independent reward function logic, including fairness and stability)
* `Tests.m` (Unit tests for channel model and reward functions using `matlab.unittest.TestCase`)
* `smokeTest.m` (Environment validation and random action smoke test)
* `trainSAC.m` (Main training script)

**【Physical Layer & Simulation Parameters (To be mapped in Config.m)】**

* `W` (RB Bandwidth): 180 kHz
* `B_RBs` (Total RBs in system): 20
* `M_mini_slots` (Mini-slots per Slot): 7
* `URLLC_Traffic`: Poisson process, $\lambda = 3$ packets/mini-slot
* `D_one` (Single URLLC packet size): 32 bytes
* `tau_req` (URLLC latency limit $\tau_u^{req}$): 0.5 ms
* `eMBB_QoS` ($r_e^{QoS}$): Randomly distributed between 20 Mbps and 50 Mbps
* `N_g` (Number of channel quality groups): 5 (Excellent, Good, Medium, Poor, Very Poor)
* `Q_levels` (Queue quantization levels): 12 ($q_0$ to $q_{11}$)
* `Reward_Weights`: $\omega_1, \omega_2, \omega_3$ (for Satisfaction, Fairness, Stability)

**【Core Communication Mathematical Formulas】**

1. **eMBB Data Rate (No URLLC Interference)**: $r_{eb}(t) = W \log_2(1 + \text{SNR}_{eb})$
2. **eMBB Data Rate (With URLLC Superposition Interference)**: Calculate SINR, treating the power allocated to URLLC as interference $I_{eu}(t,m)$.
3. **NOMA SIC Condition**: URLLC can only be successfully decoded first (SIC success) if $|h_{ub}|^2 \le |h_{eb}|^2$ and $\theta_b > 0.5$.

**【Reinforcement Learning MDP Formulation】**

The SAC agent interacts based on the following Markov Decision Process $(S, A, R, P, \gamma)$:

* **State Space ($S$)**:
At the $m$-th Mini-slot of the $t$-th Slot, the state is a 1D vector of length 11:
$s(t, m) = [\Phi(\phi_{u, G_1}), \dots, \Phi(\phi_{u, G_5}), \Phi(\phi_{e, G_1}), \dots, \Phi(\phi_{e, G_5}), m]$
* $\Phi(\phi_{u, G_k})$: Quantized queue length for URLLC group $k$ ($0 \sim 11$).
* $\Phi(\phi_{e, G_k})$: Quantized queue length for eMBB group $k$ ($0 \sim 11$).
* $m$: Current Mini-slot index ($0 \sim 6$).
* **Action Space ($A$)**:
The continuous power allocation ratio output by the agent:
$a(t, m) = [\theta_{G_1}, \theta_{G_2}, \theta_{G_3}, \theta_{G_4}, \theta_{G_5}], \quad \theta_{G_k} \in [0, 1]$
* **Reward Function ($R$)**:
$R(t, m) = \begin{cases} \omega_1 F^{sat'} + \omega_2 F^{fair'} + \omega_3 F^{sta'}, & \text{if } \tau_u^{act} \le \tau_u^{req} \\ 0.0, & \text{if } \tau_u^{act} > \tau_u^{req} \end{cases}$
* **State Transition ($P$)**:
After executing action $a(t, m)$, the environment transitions to $s(t, m+1)$ via:

1. **Queue Consumption**: Deduct the URLLC and eMBB data successfully transmitted within the Mini-slot based on $\theta$.
2. **New Packet Arrival**: Generate new URLLC packets according to a Poisson Process ($\lambda=3$) using `poissrnd` and add them to the respective group queues.
3. **Time Step**: $m \leftarrow m + 1$. When $m = 7$, the Episode terminates (`IsDone = true`).

---

## Part 2: Divide and Conquer Tasks (Detailed Engineering Specs)

### 🛠️ Task 1: Global Configuration Setup (`Config.m`)

> **Instructions:** Create a centralized configuration class named `Config.m` using MATLAB's `classdef` with `properties (Constant)` for strict enforcement.
> **Requirements:**
> 1. **Physical Layer Constants:**
> 
> 
> * `W = 180e3;` (Bandwidth per RB in Hz, i.e., 180 kHz)
> * `B_RBs = 20;` (Total Resource Blocks)
> * `M_mini_slots = 7;` (Mini-slots per eMBB Slot)
> * `Noise_Power = -114.0;` (Background noise in dBm)
> 
> 
> 2. **Traffic & QoS Parameters:**
> 
> 
> * `lambda_urllc = 3.0;` (Poisson arrival rate for URLLC)
> * `D_one = 256;` (Single URLLC packet size in bits, i.e., 32 bytes $\times$ 8)
> * `tau_req = 0.5e-3;` (URLLC latency limit in seconds, i.e., 0.5 ms)
> * `eMBB_QoS_min = 20e6;`, `eMBB_QoS_max = 50e6;` (bps)
> 
> 
> 3. **RL Hyperparameters:**
> 
> 
> * `N_g = 5;` (Number of channel quality groups)
> * `Q_levels = 12;` (Quantization levels $0 \sim 11$)
> * `phi_th = 10000.0;` (Queue threshold in bits for the maximum quantization level $q_{11}$)
> * `omega_1 = 0.5;`, `omega_2 = 0.25;`, `omega_3 = 0.25;` (Reward weights)
> 
> 
> 4. **Constraint:** Ensure no hardcoded magic numbers exist in any subsequent modules; everything MUST be imported from this class (e.g., `Config.B_RBs`).
> 
> 

### 🛠️ Task 2: 5G Physical Layer & State Reduction Module (`ChannelStateProcessor.m`)

> **Instructions:** Implement functions to handle physical channel simulations and state space dimensionality reduction.
> **Requirements:**
> 1. **`snrToMCS(snrDB)`**: Implement a mapping from SNR (in dB) to a 5G NR MCS Index (0 to 28). You may leverage 5G Toolbox functions like `nrPerfectCQI` as a reference. If SNR is too low, return 0.
> 2. **`groupUsers(mcsArray)`**: Divide users into `N_g=5` groups based on their MCS index. Use these fixed bins: Group 1 (MCS 24-28), Group 2 (MCS 18-23), Group 3 (MCS 12-17), Group 4 (MCS 6-11), Group 5 (MCS 0-5). Return a cell array or struct mapping group ID to user indices.
> 3. **`quantizeQueue(queueLength)`**: Implement uniform interval quantization.
> 
> 
> * Formula: $q = \lfloor \text{queue\_length} / (\text{phi\_th} / (\text{Q\_levels} - 2)) \rfloor + 1$
> * If `queueLength == 0`, return `0`.
> * If `queueLength >= phi_th`, clamp the return value to `11`.
> 
> 
> 4. **Unit Testing:** Write robust test functions in `Tests.m` using `matlab.unittest.TestCase`. Mock SNR arrays to assert that `groupUsers` correctly routes users, and test `quantizeQueue` with inputs `0`, `phi_th / 2`, and `phi_th * 10` to ensure no out-of-bounds index errors.
> 
> 

### 🛠️ Task 3: RL Custom Environment Setup (`RANSlicingEnv.m` init)

> **Instructions:** Inherit from `rl.env.MATLABEnvironment` to implement the `RANSlicingEnv` class. Focus strictly on the constructor `RANSlicingEnv()` and the `reset()` method.
> **Requirements:**
> 1. **Space Definitions (Constructor):**
> 
> 
> * Define `actInfo = rlNumericSpec([5 1], 'LowerLimit', 0, 'UpperLimit', 1);`
> * Define `obsInfo = rlNumericSpec([11 1], 'LowerLimit', 0, 'UpperLimit', 11);` (Indices 1-5 for URLLC queues, 6-10 for eMBB queues, index 11 for Mini-slot $m$ spanning $0 \sim 6$).
> 
> 
> 2. **Reset Logic:** `reset(this)` must initialize a new eMBB Slot (1 ms duration).
> 
> 
> * Generate independent Rayleigh fading channel gains for eMBB ($h_{eb}$) and URLLC ($h_{ub}$) users.
> * *Mathematical Implementation:* $h = (x + iy) / \sqrt{2}$, where $x, y \sim \mathcal{N}(0, 1)$ using `randn()`. (Alternatively, integrate `nrTDLChannel` from 5G Toolbox).
> * Assign random $r_e^{QoS}$ to eMBB users uniformly using `rand()`.
> * Reset $m=0$, clear historical rate buffers, and return `InitialObservation`.
> 
> 

### 🛠️ Task 4: Reward Function & Multi-Objective Optimization (`calculateReward.m`)

> **Instructions:** Implement a pure mathematical function `calculateReward(actualDelay, reqDelay, embbRates, embbQoS)`.
> **Requirements:**
> 1. **Hard Constraint (Penalty):** `if actualDelay > reqDelay; reward = 0.0; return; end`
> 2. **eMBB Satisfaction ($F^{sat'}$):** Calculate element-wise ratio $x_i = \min(1.0, r_i / QoS_i)$. Then $F^{sat'} = \text{mean}(x)$. Handle zero division safely using `eps`.
> 3. **Jain's Fairness Index ($F^{fair'}$):**
> Implement: $F^{fair'} = (\sum x_i)^2 / (N \sum x_i^2)$.
> *Critical Edge Case:* If all rates are 0, return `0.0`.
> 4. **Stability ($F^{sta'}$):**
> Calculate the Coefficient of Variation (CV) of data rates.
> Formula: $\mu^{std} / \mu^{mean}$. Map it via $F^{sta'} = \max(0.0, 1.0 - \mu^{std} / \mu^{mean})$.
> 5. **Composite Reward:** Return $\omega_1 F^{sat'} + \omega_2 F^{fair'} + \omega_3 F^{sta'}$.
> 6. **Unit Testing:** Add test cases to `Tests.m` using `matlab.unittest`. Feed an array of `zeros` to ensure the Fairness Index function does not crash.
> 
> 

### 🛠️ Task 5: Environment Step Logic & Physical Action Mapping (`RANSlicingEnv.m` step)

> **Instructions:** Implement the `step(this, Action)` method for `RANSlicingEnv`.
> **Requirements:**
> 1. **Receive Action:** The `Action` is a vector $a = [\theta_{G_1}, \theta_{G_2}, \theta_{G_3}, \theta_{G_4}, \theta_{G_5}]$. Fetch the total burst URLLC payload $D_u^{req}$.
> 2. **RB Priority Sorting:** Sort all $B_{RBs}$ descendingly by their URLLC channel gain $|h_{ub}(t)|^2$ using `sort(..., 'descend')`.
> 3. **Physical Filtering & Clamping (NOMA SIC Rules):** Iterate through the sorted RBs. Extract $\theta_{raw} = \theta_{G_k}$.
> 
> 
> * **Case A (Mandatory Puncturing):** If $|h_{ub}(t)|^2 > |h_{eb}(t)|^2$:
> * If $\theta_{raw} \ge 0.5 \rightarrow$ Force $\theta_b = 1.0$.
> * If $\theta_{raw} < 0.5 \rightarrow$ Force $\theta_b = 0.0$.
> 
> 
> * **Case B (Superposition Permitted):** If $|h_{ub}(t)|^2 \le |h_{eb}(t)|^2$:
> * If $\theta_{raw} > 0 \rightarrow \theta_b = \max(0.51, \theta_{raw})$.
> * If $\theta_{raw} == 0 \rightarrow \theta_b = 0.0$.
> 
> 
> 
> 
> 4. **Throughput Deduction:** Calculate URLLC capacity using $\theta_b$ and deduct bits from $D_u^{req}$. If $\theta_b > 0$, calculate new eMBB SINR and update rate.
> 
> 
> * **Early Stopping:** If $D_u^{req} \le 0$, `break` the loop.
> 
> 
> 5. **Calculate Reward:** Call `calculateReward` if loop finishes.
> 6. **State Update:** Quantize queues. Increment $m$. If $m == 7$, set `IsDone = true`.
> 7. **LoggedSignals:** Return environment details via the `LoggedSignals` struct (MATLAB equivalent of the `info` dictionary), containing `urllc_actual_delay`, `embb_satisfaction`, `embb_fairness`, and `is_urllc_failed`.
> 
> 

### 🛠️ Task 6: Environment API Validation & Smoke Test (`smokeTest.m`)

> **Instructions:** Write a comprehensive smoke test script.
> **Requirements:**
> 1. **RL Toolbox API Check:** Instantiate `env = RANSlicingEnv();` and pass it to `validateEnvironment(env)`. This built-in MATLAB function strictly checks observation/action dimensions and boundaries.
> 2. **Random Agent Smoke Test:** Write a `for` loop that runs for 140 steps. In each step:
> 
> 
> * Sample random action: `action = rand(5,1);` (or use internal sampling logic).
> * Step environment: `[obs, reward, isdone, logged] = step(env, action);`
> * Print `reward` and `logged.is_urllc_failed` using `disp()`.
> * If `isdone` is true, call `reset(env)`.
> 
> 
> 3. **Assertion:** Ensure no index out-of-bounds, `NaN`, or divide-by-zero errors occur.
> 
> 

### 🛠️ Task 7: RL Training Script (`trainSAC.m`)

> **Instructions:** Write the main execution script `trainSAC.m`.
> **Requirements:**
> 1. **Initialization:** `env = RANSlicingEnv();`. Fix seed using `rng(42)`.
> 2. **SAC Model Configuration:** > * Build SAC agent using `agent = rlSACAgent(env.getObservationInfo, env.getActionInfo);`.
> 
> 
> * Adjust `agent.AgentOptions.MiniBatchSize` and learning rates.
> 
> 
> 3. **Execution & Checkpoints:**
> 
> 
> * Setup `trainOpts = rlTrainingOptions('MaxEpisodes', ..., 'MaxStepsPerEpisode', 7, 'Plots', 'training-progress');` (This replaces TensorBoard).
> * Setup `trainOpts.SaveAgentCriteria = 'EpisodeCount';` and `trainOpts.SaveAgentValue = 500;` to save weights to a `./models/` folder.
> * Execute `trainingStats = train(agent, env, trainOpts);`.
> 
> 
