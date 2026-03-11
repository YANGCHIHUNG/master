# Iterative Task: Implement 3GPP Standard CQI/MCS Link Abstraction

## 🎯 Task Overview
The current environment calculates throughput using the ideal Shannon Capacity formula `W * log2(1 + SINR)`. This mathematically overestimates realistic 5G network capacity. 
This task will replace the ideal Shannon Capacity with a realistic 3GPP Link Abstraction using standardized Channel Quality Indicator (CQI) and Spectral Efficiency (SE) look-up tables based on TS 38.214.

**Development Guidelines**: Strictly execute from Phase 1 to Phase 3. Use `matlab -batch` for Check Points and fix any errors before proceeding.

---

## 🟢 Phase 1: Overhaul `ChannelStateProcessor.m` with 3GPP Tables
**Goal: Implement accurate mapping from SINR -> CQI -> Spectral Efficiency (SE).**

- [ ] **Step 1.1: Replace `snrToMCS` with `snrToCQI`**
  - Open `ChannelStateProcessor.m`.
  - Delete the `snrToMCS` method.
  - Create a new static method `cqi = snrToCQI(snrDb)`.
  - Implement a look-up logic using the following 3GPP threshold approximations (for 10% BLER):
    ```matlab
    % Thresholds for CQI 1 to 15 (in dB)
    thresholds = [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7];
    ```
    *(Logic: If snrDb < -6.7, cqi = 0. If snrDb >= 22.7, cqi = 15. Otherwise, find the highest CQI level where snrDb >= threshold.)*
- [ ] **Step 1.2: Add Spectral Efficiency Mapping `cqiToEfficiency`**
  - Create a new static method `efficiency = cqiToEfficiency(cqi)`.
  - Implement a mapping array matching CQI 0 to 15 to their spectral efficiencies (bps/Hz):
    ```matlab
    % Index 1 corresponds to CQI 0, Index 16 corresponds to CQI 15
    seTable = [0.0, 0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141, 2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547];
    efficiency = seTable(cqi + 1);
    ```
- [ ] **Step 1.3: Update `groupUsers` for CQI Range**
  - Modify `groupUsers` to accept `cqiPerRB` instead of `mcsPerRB`.
  - Change the grouping edges to fit the 0-15 CQI range evenly into `Config.N_g` (5) groups.
  - Replace the old edges `[0, 5, 10, 15, 20, 28]` with new edges `[0, 3, 6, 9, 12, 15]`.

> 🛑 **Phase 1 Check Point:**
> Run `matlab -batch "ChannelStateProcessor.snrToCQI(20)"`. It should output `13`.
> Run `matlab -batch "ChannelStateProcessor.cqiToEfficiency(15)"`. It should output `5.5547`.

---

## 🟢 Phase 2: Integrate 3GPP Mapping into `RANSlicingEnv.m`
**Goal: Calculate physical rates using the new Spectral Efficiency mappings instead of log2.**

- [ ] **Step 2.1: Fix Pre-loop Grouping**
  - Open `RANSlicingEnv.m` and locate the `step` method.
  - Find the pre-loop grouping logic: `mcsPerRB = ChannelStateProcessor.snrToMCS(embbSNRdB);`
  - Change it to: `cqiPerRB = ChannelStateProcessor.snrToCQI(embbSNRdB);`
  - Pass `cqiPerRB` to `groupUsers`.
- [ ] **Step 2.2: Replace URLLC Shannon Rate Calculation**
  - Inside the `for idx = 1:rbCount` loop, find the URLLC rate calculation:
    `urllcRate = Config.W * log2(1 + urllcSINR);`
  - Replace it by converting `urllcSINR` to dB, finding CQI, and getting efficiency:
    ```matlab
    urllcSINRdB = 10 * log10(max(eps, urllcSINR));
    urllcCQI = arrayfun(@ChannelStateProcessor.snrToCQI, urllcSINRdB);
    urllcRate = Config.W * arrayfun(@ChannelStateProcessor.cqiToEfficiency, urllcCQI);
    ```
- [ ] **Step 2.3: Replace eMBB Shannon Rate Calculation**
  - Similarly, find the eMBB rate calculation:
    `embbRatesPerRB(rb) = Config.W * log2(1 + embbSINR);`
  - Replace it with:
    ```matlab
    embbSINRdB = 10 * log10(max(eps, embbSINR));
    embbCQI = ChannelStateProcessor.snrToCQI(embbSINRdB);
    embbRatesPerRB(rb) = Config.W * ChannelStateProcessor.cqiToEfficiency(embbCQI);
    ```

> 🛑 **Phase 2 Check Point:**
> Run `matlab -batch "smokeTest"`. Ensure 140 steps complete without index or undefined function errors.

---

## 🟢 Phase 3: Final Diagnostics
**Goal: Verify the agent interacts with the new hard-capped capacity environment.**

- [ ] **Step 3.1: Execute `testAgent.m`**
  - Run `matlab -batch "testAgent"`.
  - The script should run smoothly and output the episode metrics. Because the throughput is now strictly capped by CQI 15 (max 5.55 bps/Hz instead of infinite), the environment is now 100% physically realistic.

> 🛑 **Final Check Point (Phase 3):**
> If `testAgent` outputs successful logs with the new 3GPP capacities, mark this iterative task as fully complete!