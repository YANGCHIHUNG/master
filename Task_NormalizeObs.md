# Iterative Task: Normalize Observation Space for Stable RL Training

## 🎯 Task Overview
The introduction of the FTP3 bursty traffic model generates eMBB queue sizes in the millions (e.g., 15,000,000 bits). Feeding these unnormalized raw values into the SAC neural networks causes immediate gradient explosion and "Dying ReLU", forcing the Q-values to remain stuck at 0. 
This task will normalize the state observation vector to keep neural network inputs roughly within a small numerical range (e.g., [0, 1]).

**Development Guidelines**: Execute Phase 1 below and use `matlab -batch "smokeTest"` as the Check Point.

---

## 🟢 Phase 1: Normalize Observation Vector in `RANSlicingEnv.m`
**Goal: Scale down queues and indices before returning the observation vector.**

- [ ] **Step 1.1: Locate the Observation Assembly**
  - Open `RANSlicingEnv.m` and find the `getObservation(this)` method (or the logic where the `obs` array is built).
- [ ] **Step 1.2: Apply Normalization Scaling**
  - Before assembling and returning the `obs` array, scale down the raw physical components using temporary variables:
    - Scale `URLLCGroupQueues` by dividing by `10000.0` (approximate max URLLC backlog).
    - Scale `eMBBGroupQueues` by dividing by `1e7` (10 million, to compress FTP3 bursts).
    - Scale `MiniSlotIndex` by dividing by `Config.M_mini_slots`.
  - Ensure that the returned `obs` vector uses these scaled temporary variables instead of the raw `this.URLLCGroupQueues` and `this.eMBBGroupQueues`.
  - *(Critical Note: Do not alter the actual internal state variables `this.eMBBGroupQueues` etc.; only scale the copies used to construct the output `obs`.)*

> 🛑 **Phase 1 Check Point:**
> Run `matlab -batch "smokeTest"`. Ensure it completes without state dimension or numeric errors.