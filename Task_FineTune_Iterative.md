# Iterative Task: Fine-Tune Path Loss and QoS for Spatial Diversity

## 🎯 Task Overview
The trained agent successfully achieved 0% URLLC failure, but eMBB satisfaction remains low. This is because the average SNR (~24 dB) is still mapping all 20 RBs to Group 1 (`[20 0 0 0 0]`), forcing the agent to apply high NOMA interference globally.
We need to increase the Path Loss to spread the RBs across multiple groups (e.g., Groups 2, 3, and 4) and slightly lower the eMBB QoS to match the reduced capacity.

---

## 🟢 Phase 1: Fine-Tune `Config.m`
**Goal: Shift the average SNR to ~14 dB to populate middle groups.**

- [x] **Step 1.1: Increase Path Loss**
  - Open `Config.m`.
  - Change `Path_Loss_dB` from `120.0` to `130.0`.
- [x] **Step 1.2: Scale down eMBB QoS Targets**
  - Change `eMBB_QoS_min` from `500e3` to `200e3` (200 kbps).
  - Change `eMBB_QoS_max` from `1.5e6` to `800e3` (800 kbps).

> 🛑 **Phase 1 Check Point:**
> Run `matlab -batch "Config.Path_Loss_dB, Config.eMBB_QoS_max"` to verify the values are 130 and 800000.

---

## 🟢 Phase 2: Verify Diversity via Diagnostic Run
**Goal: Ensure RBs are no longer bunched in Group 1.**

- [x] **Step 2.1: Run `testAgent.m`**
  - Execute `matlab -batch "testAgent"`.
  - Observe the `Active eMBB Groups` in the log. You should now see numbers spread across the middle indices (e.g., `[2 5 8 4 1]`).
  - *Note: Since the agent was trained on the old 120dB environment, its actions might look a bit confused in this new 130dB environment, which is normal. We only care that the Active Groups are distributed.*

> 🛑 **Final Check Point (Phase 2):**
> If the RBs are successfully distributed across multiple groups in the log, mark this task as complete!
