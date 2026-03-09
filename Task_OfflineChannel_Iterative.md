# Iterative Development Task Specification: Offline 5G TDL Channel Trace Generation and Environment Integration

## 🎯 Task Overview
The objective of this task is to replace the channel gains originally generated using pure randomness (`randn`) in `RANSlicingEnv.m` with an offline-generated TDL-A channel trace file using the MATLAB 5G Toolbox.
**Development Guidelines**: Please strictly follow the sequence from Phase 1 to Phase 4. Execute the "Check Point" at the end of each phase, and proceed to the next phase only after confirming no errors occurred.

---

## 🟢 Phase 1: Develop Offline Trace Generation Script (`generate_channel_trace.m`)
**Goal: Create an independent script capable of generating a `.mat` trace file.**

- [x] **Step 1.1: Script Initialization and Parameter Setup**
  - Create `generate_channel_trace.m` in the project root directory.
  - Read `B_RBs` and `M_mini_slots` from `Config.m`.
  - Set the total number of Mini-slots to generate to `100,000` steps.
- [x] **Step 1.2: Establish 5G TDL Channel Model**
  - Use `nrTDLChannel` to instantiate the model, setting `DelayProfile = 'TDL-A'`.
  - Set `DelaySpread = 30e-9` and `MaximumDopplerShift = 10`.
  - Configure the corresponding `nrCarrierConfig` (SubcarrierSpacing = 15) and sample rate.
- [x] **Step 1.3: Write Loop to Generate Time-Frequency Resource Grid and Average**
  - Use an outer loop running iteratively per Slot.
  - Use a dummy time-domain waveform and `nrPerfectChannelEstimate` to obtain the channel response matrix.
  - Average the $|h|^2$ of every 12 subcarriers (1 RB) in the frequency domain, and map the time-domain Symbols to the corresponding Mini-slots.
- [x] **Step 1.4: Generate Dual Traces and Save**
  - Assign the generated matrix to `urllcChannelTrace`.
  - Use `circshift` (e.g., shift by 5 steps) to offset the original matrix to create `embbChannelTrace`, simulating independent fading for a different user.
  - Use `save('ChannelTraces.mat', 'urllcChannelTrace', 'embbChannelTrace');` to save the file.

> 🛑 **Phase 1 Check Point:**
> Execute the `generate_channel_trace` script in the terminal.
> Verify that `ChannelTraces.mat` is successfully generated in the workspace, and after loading, both variables have a size of `[20, 100000]` (assuming `Config.B_RBs` is 20).

---

## 🟢 Phase 2: Environment Properties and Constructor Expansion (`RANSlicingEnv.m`)
**Goal: Equip the environment with the ability to load and store trace data upon instantiation.**

- [x] **Step 2.1: Declare Private Properties**
  - Add 4 variables in the `properties (Access = private)` block of `RANSlicingEnv.m`:
    - `TraceURLLC`
    - `TraceeMBB`
    - `GlobalStepIndex`
    - `MaxTraceSteps`
- [x] **Step 2.2: Modify Constructor to Load File**
  - Inside `function this = RANSlicingEnv()`, immediately below `this = this@rl.env.MATLABEnvironment(...)`:
  - Add `traceData = load('ChannelTraces.mat');`.
  - Assign the loaded data to `this.TraceURLLC` and `this.TraceeMBB` respectively.
  - Calculate and store the number of columns of the matrix into `this.MaxTraceSteps`.
  - Use `randi([1, this.MaxTraceSteps])` to randomly initialize `this.GlobalStepIndex`.

> 🛑 **Phase 2 Check Point:**
> Execute `env = RANSlicingEnv();` in the MATLAB Command Window.
> Only proceed to Phase 3 if the environment is successfully instantiated without any errors.

---

## 🟢 Phase 3: Modify Environment Reset Logic (`reset` method)
**Goal: Ensure that at the start of each Episode, the channel is read from the trace rather than randomly generated.**

- [x] **Step 3.1: Remove Old Random Logic**
  - In `reset(this)`, find the 4 lines of code that originally calculated `hURLLC`, `heMBB`, and their `abs(...).^2`, and delete or comment them out.
- [x] **Step 3.2: Implement Circular Buffer Read Logic**
  - At the deleted location, add the circular index calculation: `safeIndex = mod(this.GlobalStepIndex - 1, this.MaxTraceSteps) + 1;`
- [x] **Step 3.3: Assignment**
  - Set `this.URLLCChannelGain = this.TraceURLLC(:, safeIndex);`
  - Set `this.eMBBChannelGain = this.TraceeMBB(:, safeIndex);`

> 🛑 **Phase 3 Check Point:**
> Execute `obs = reset(env);` in the command line.
> Check if no errors occur and verify that the environment successfully resets using the offline trace data.

---

## 🟢 Phase 4: Modify Environment Step Logic (`step` method) and Final Testing
**Goal: Advance the trace synchronously during the environment step and verify smooth system operation.**

- [x] **Step 4.1: Update Global Step Index**
  - Near the end of the `step()` method, find the location of `this.MiniSlotIndex = this.MiniSlotIndex + 1;`.
  - Immediately below it, add `this.GlobalStepIndex = this.GlobalStepIndex + 1;`.
- [x] **Step 4.2: Update Channel in `~isDone` Block**
  - Find the `if ~isDone` conditional block.
  - Delete or comment out the original `hURLLC` and `heMBB` generation logic.
  - Paste the identical read logic from Phase 3:
    ```matlab
    safeIndex = mod(this.GlobalStepIndex - 1, this.MaxTraceSteps) + 1;
    this.URLLCChannelGain = this.TraceURLLC(:, safeIndex);
    this.eMBBChannelGain = this.TraceeMBB(:, safeIndex);
    ```

> 🛑 **Final Check Point (Phase 4):**
> 1. Execute the existing `smokeTest.m` in the terminal.
> 2. If the smoke test successfully completes all 140 steps without throwing any `NaN`, `Inf`, or "Index out of bounds" errors, this iterative development task is successfully completed!
