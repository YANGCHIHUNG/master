# Iterative Development Task Specification: Fix Physical Layer Parameters & QoS Demands

## 🎯 Task Overview
The current environment suffers from two physical anomalies: 
1) The SNR is unrealistically high (resulting in all users being mapped to MCS Group 1) because the system lacks Path Loss modeling. 
2) The eMBB QoS demand per RB (10 Mbps ~ 30 Mbps) far exceeds the Shannon capacity limit for a 180 kHz bandwidth.
This task will introduce Tx Power and Path Loss to correct the SNR, and scale down the eMBB QoS targets.

**Development Guidelines**: Please strictly follow the sequence from Phase 1 to Phase 3. Execute the "Check Point" at the end of each phase using headless MATLAB (`matlab -batch`), and proceed only if no errors occur.

---

## 🟢 Phase 1: Update Physical Constants in `Config.m`
**Goal: Introduce transmission power, path loss, and realistic QoS limits.**

- [x] **Step 1.1: Add Tx Power and Path Loss**
  - Open `Config.m`.
  - In the "Physical layer constants" block, add two new properties:
    - `Tx_Power_dBm (1,1) double = 30.0`
    - `Path_Loss_dB (1,1) double = 120.0`
- [x] **Step 1.2: Adjust eMBB QoS Range**
  - In the "Traffic and QoS parameters" block, find `eMBB_QoS_min` and `eMBB_QoS_max`.
  - Modify their values to mathematically realistic targets for a 180kHz RB:
    - Change `eMBB_QoS_min` to `500e3` (representing 500 kbps).
    - Change `eMBB_QoS_max` to `1.5e6` (representing 1.5 Mbps).

> 🛑 **Phase 1 Check Point:**
> Run `matlab -batch "Config.Tx_Power_dBm, Config.eMBB_QoS_min"` in the terminal.
> Ensure it outputs 30 and 500000 respectively without any undefined property errors.

---

## 🟢 Phase 2: Apply Path Loss to Channel Gains in `RANSlicingEnv.m`
**Goal: Scale the raw channel gains using the new Tx Power and Path Loss parameters before calculating SNR.**

- [x] **Step 2.1: Define Receive Power Factor**
  - Open `RANSlicingEnv.m` and locate the `step` method.
  - Find the block where `noisePowerLinear` is defined (around line 105).
  - Immediately below `noisePowerLinear = 10 ^ (Config.Noise_Power / 10);`, add the following calculation for the receive power factor:
    `rxPowerFactor = 10 ^ ((Config.Tx_Power_dBm - Config.Path_Loss_dB) / 10);`
- [x] **Step 2.2: Fix Pre-loop SNR Calculation (For MCS Grouping)**
  - Find the line: `embbSNRLinear = this.eMBBChannelGain ./ max(noisePowerLinear, eps);`
  - Modify it to multiply the channel gain by the new factor:
    `embbSNRLinear = (this.eMBBChannelGain .* rxPowerFactor) ./ max(noisePowerLinear, eps);`
- [x] **Step 2.3: Fix In-loop Channel Gains (For NOMA/SIC and Rate Calculation)**
  - Scroll down inside the `for idx = 1:rbCount` loop.
  - Find the variable assignments for `hub` and `heb`:
    ```matlab
    hub = this.URLLCChannelGain(rb);
    heb = this.eMBBChannelGain(rb);
    ```
  - Modify them to include the `rxPowerFactor`:
    ```matlab
    hub = this.URLLCChannelGain(rb) * rxPowerFactor;
    heb = this.eMBBChannelGain(rb) * rxPowerFactor;
    ```
  - *(Note: By redefining `hub` and `heb` here, the subsequent NOMA logic and SINR formulas will automatically use the correct scaled values. Do not alter the complex SINR mathematical formulas below this.)*

> 🛑 **Phase 2 Check Point:**
> Run `matlab -batch "smokeTest"` in the terminal.
> Ensure the script completes all 140 steps without throwing any errors or warnings.

---

## 🟢 Phase 3: Verify Metrics via Diagnostic Test
**Goal: Run a single episode to observe if the user groups and satisfaction rates are now distributed logically.**

- [x] **Step 3.1: Execute `testAgent.m`**
  - Run `matlab -batch "testAgent"` in the terminal.
  - Inspect the output. You should now observe:
    1. `Action (theta)` values and `eMBB Queues` should be populated across multiple indices (not just index 1), proving users are now distributed across Group 1 to Group 5.
    2. The `Average eMBB Satisfaction` at the end of the episode should be significantly higher than 1% (likely in the 50%~90% range), proving the capacity demands are now physically solvable.

> 🛑 **Final Check Point (Phase 3):**
> If the testAgent outputs logical values across multiple groups and doesn't crash, mark this task as fully complete!
