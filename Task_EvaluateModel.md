# Task Specification: Implement Monte Carlo Evaluation and Visualization (`evaluateModel.m`)

## 🎯 Task Overview
The current `testAgent.m` only evaluates a single episode, which is insufficient for academic or statistical proof of reliability. 
This task will create a new script `evaluateModel.m` that performs a Monte Carlo simulation over 500 episodes, aggregates step-by-step metrics, and generates publication-ready plots (e.g., CDF of URLLC Delays, Histogram of eMBB Satisfaction).

**Development Guidelines**: Execute from Phase 1 to Phase 3. Use `matlab -batch "evaluateModel"` for the final Check Point.

---

## 🟢 Phase 1: Script Scaffolding and Auto-Loader
**Goal: Create `evaluateModel.m` and implement robust agent loading.**

- [ ] **Step 1.1: Create File and Initialize**
  - Create `evaluateModel.m` in the root directory.
  - Set `numEpisodes = 500;` at the top of the script.
  - Initialize the environment `env = RANSlicingEnv();`.
- [ ] **Step 1.2: Auto-Load Latest Agent**
  - Implement logic to look into the directory specified by `Config.train_save_dir`.
  - Find all `Agent*.mat` files. If none exist, throw an error.
  - Sort by modification date and load the newest one into a variable named `agent`. (Extract it from `loadedData.saved_agent`).

---

## 🟢 Phase 2: Monte Carlo Simulation Loop
**Goal: Run 500 episodes and collect granular step-by-step metrics.**

- [ ] **Step 2.1: Preallocate Data Arrays**
  - Preallocate arrays to store data across all steps of all episodes:
    - `allUrllcDelays = zeros(numEpisodes * Config.M_mini_slots, 1);`
    - `allEmbbSatisfactions = zeros(numEpisodes * Config.M_mini_slots, 1);`
    - `allUrllcFailures = 0;`
  - Maintain a counter `globalStep = 1;`.
- [ ] **Step 2.2: Execution Loop**
  - Create a `for ep = 1:numEpisodes` loop.
  - Inside, call `obs = reset(env);`
  - Create an inner loop `for k = 1:Config.M_mini_slots`.
  - Use `actionCell = getAction(agent, obs); action = actionCell{1};`.
  - Call `[obs, ~, isDone, logged] = step(env, action);`.
- [ ] **Step 2.3: Data Logging**
  - Inside the inner loop, record:
    - `allUrllcDelays(globalStep) = logged.urllc_actual_delay * 1000;` (Convert to ms)
    - `allEmbbSatisfactions(globalStep) = logged.embb_satisfaction * 100;` (Convert to %)
    - `allUrllcFailures = allUrllcFailures + logged.is_urllc_failed;`
    - Increment `globalStep = globalStep + 1;`.
    - `if isDone, break; end`

---

## 🟢 Phase 3: Statistical Summary & Visualization
**Goal: Print a console summary and generate a `.png` figure with academic plots.**

- [ ] **Step 3.1: Print Console Summary**
  - Calculate and `fprintf` the following metrics to the console:
    - Total Episodes tested.
    - Overall URLLC Failure Rate (%).
    - 99th Percentile URLLC Delay (ms) using `prctile(allUrllcDelays, 99)`.
    - Average eMBB Satisfaction (%).
- [ ] **Step 3.2: Generate Plots**
  - Create a `figure('Name', 'Evaluation Results', 'Position', [100, 100, 1200, 400]);`.
  - **Subplot 1 (1,2,1): CDF of URLLC Delays**
    - Use `ecdf(allUrllcDelays)` or calculate the CDF manually. Plot the CDF curve.
    - Add a vertical red dashed line at `Config.tau_req * 1000` (Target Constraint).
    - Set title 'CDF of URLLC Delay', xlabel 'Delay (ms)', ylabel 'Probability'.
  - **Subplot 2 (1,2,2): Histogram of eMBB Satisfaction**
    - Use `histogram(allEmbbSatisfactions, 'Normalization', 'probability')`.
    - Set title 'Distribution of eMBB Satisfaction', xlabel 'Satisfaction (%)', ylabel 'Probability'.
- [ ] **Step 3.3: Save Figure**
  - Add `saveas(gcf, 'Evaluation_Report.png');` at the end of the script to ensure the plot is saved when running headlessly.

> 🛑 **Final Check Point:**
> Run `matlab -batch "evaluateModel"` in the terminal.
> Ensure the script completes 500 episodes, prints the summary, and successfully saves `Evaluation_Report.png` to the root directory without any errors.