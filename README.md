# 5G RAN Slicing RL Simulator (MATLAB)

這個專案是一個研究型的 MATLAB 模擬器，用於在 5G NR 無線接入網（RAN）中，使用強化學習（Soft Actor-Critic, SAC）學習在 URLLC 與 eMBB 服務間做資源切片（PRB puncturing / quota）分配。

以下說明包含系統概觀、5G PHY 與 channel 模組、以及專案中 MDP（馬可夫決策過程）的明確定義，方便論文撰寫、複現實驗與教學示範。

----

## 主要內容
- RANSlicingEnv.m: 自訂 RL environment（遵循 MATLAB RL Toolbox API）
- NRPhyChannel.m: 用 5G Toolbox TDL channel + CQI → MCS → TBS 的 PHY 抽象
- calculateReward.m: 所有 reward 組成（URLLC 延遲懲罰、eMBB backlog 懲罰、Jain fairness 激勵）
- trainSAC.m: 建立並訓練 SAC 代理的腳本
- smokeTest.m / Tests.m: 進行環境與單元測試
- Config.m: 集中式實驗與超參數（觀測正規化、reward scaling、PHY/traffic 參數）

模型檔案與訓練紀錄存放在 `models/`（Agent checkpoints 與 trainingStats）。

----

## 需要的 MATLAB 版本與 Toolbox（建議）
- MATLAB R2021a 或更新
- Reinforcement Learning Toolbox
- 5G Toolbox
- Communications Toolbox
- Signal Processing Toolbox（視需要）

若在 CI 或不同機器上執行，請確保已安裝相同版本的 toolbox 或在 README 備註版本。

----

## 5G 環境（PHY & Channel）概覽

- Carrier: FR1, 30 kHz SCS, 20 PRB study grid（由 `Config` 控制）
- Channel: 3GPP TR 38.901 TDL-C，支援 pathloss、shadow fading、Doppler（`NRPhyChannel.createTDLChannel`）
- CQI 與 LA: 透過 wideband CQI 與 CQI→MCS→TBS 映射計算每群組可服務位元（`NRPhyChannel.mapCQIToMCS`、`calculateTBS`）
- 群組化: 以 eMBB PRB CQI 分群（`ChannelStateProcessor.groupUsers`），群群大小為 `Config.N_g`
- mini-slot 模型: 每 slot 被切分為 `Config.M_mini_slots` mini-slots，URLLC 可能會在 mini-slot 被 puncture 到 PRB

設計重點：精準 PHY 模式用於最終評估；若需要大量訓練可加入簡化（fast）PHY 模式以加速樣本收集。

----

## MDP（馬可夫決策過程）定義

環境與 agent 的接口遵循 MATLAB 的 rl.env.MATLABEnvironment 規範。

- State (observation): column vector of size `2*Config.N_g + 3`
  - normalized URLLC group queues (Config.N_g × 1)
  - normalized eMBB group queues (Config.N_g × 1)
  - normalized mini-slot index (scalar)
  - URLLC wideband CQI (scalar)
  - eMBB wideband CQI (scalar)

  正規化採用 log10(1 + queue) 並除以 `Config.obs_urlllc_log_scale` / `Config.obs_embb_log_scale` 以避免 state 飽和。

- Action: continuous vector in [0, 1]^N_g
  - 每個維度代表對應 group 的 URLLC puncturing quota θ_Gk，對應要分配給 URLLC 的 PRB 比例（在該群內 PRB 數的基礎上取整數）。

- Reward: scalar，組成如下（見 `calculateReward.m`）：
  - URLLC delay soft penalty: 若 actualDelay > τ_req，給負值比例懲罰
  - eMBB backlog penalty: -Config.embb_penalty_scale * sum(queueBits)，並 clip 下界（避免極大負值）
  - Jain fairness 小幅正向獎勵：只對 active eMBB groups（有 backlog 的群組）計算
  - 最終使用 `Config.reward_fairness_weight` 調整 fairness 權重，並用 `Config.reward_scale` 做總尺度分割（避免 SAC 網路梯度爆炸）

- Transition: 按 mini-slot 推進；環境每 step 執行：
  1) eMBB arrivals（Pareto-based FTP3 模擬，受 `lambda_embb` 控制）
  2) PHY 層計算每群所能提供的 URLLC / eMBB TBS（根據 CQI 與 action 指定的 puncturing）
  3) 服務 URLLC 與 eMBB，更新 queues
  4) 計算 reward，輸出 observation 與 loggedSignals

- Episode: 一個 episode 為一個 Slot 的多個 mini-slot，長度由 `Config.Max_Episode_Steps` 決定（通常等於 slot × mini-slots）。

----

## 重要 `Config` 參數（實驗者需注意）

- Observation normalization:
  - `Config.obs_urlllc_log_scale`（預設 6.0）
  - `Config.obs_embb_log_scale`（預設 9.0）
- Reward shaping:
  - `Config.reward_fairness_weight`（預設 0.1）
  - `Config.reward_scale`（預設 100.0）
- PHY / channel / traffic / RL hyperparameters都集中在 `Config.m`，請在每次實驗前檢閱並存檔（推薦與模型 checkpoint 一起儲存）。

----

## 如何執行（快速上手）

1. 在 MATLAB 中開啟專案資料夾（或在 terminal 中使用 `matlab -batch`）：

```bash
matlab -batch "cd('<path-to-repo>'); smokeTest"
```

2. 執行單元測試：

```bash
matlab -batch "cd('<path-to-repo>'); results = runtests('Tests.m'); disp(results);"
```

3. 執行訓練（會在 `Config.train_save_dir` 下存檔）：

```matlab
trainSAC
```

4. 評估模型（負載敏感測試）:

```matlab
evaluateLoad
```

注意：若在非互動模式（batch）執行，建議把 plotting 關閉或把 `trainSAC` 中的 `Plots` 設為 `none`。

----

## 可重現性建議

- 在啟動訓練前，設定 `rng(Config.random_seed)` 並在儲存 checkpoint 時同時存入該 seed 與 `Config` 的 snapshot。
- 若要在不同機器/環境（或 CI）重現結果，請同時列出 `ver` 輸出（toolbox 版本）與 MATLAB 版本。

----

## 開發者筆記與進階建議

- 若要快速大量生成訓練樣本，可加入一個「fast PHY 模式」，使用 Shannon approximation 或簡化 TBS mapping 來取代 5G Toolbox 的完整 channel 模擬。
- 把所有實驗超參數（包括 `obs_urlllc_log_scale`、`reward_scale`）與模型 metadata（MATLAB 版本、seed、Config snapshot）一起存檔，便於結果追溯。

----

如果你想，我可以：
- 幫你把 README 中提到的「儲存 metadata」程式碼片段加入 `trainSAC.m`，或
- 實作 RNG 注入（讓 `RANSlicingEnv.reset(seed)` 可接受外部 seed），或
- 幫你把 README 翻成英文版本以便投稿或分享。

----

作者／維護者: YANGCHIHUNG
（自動生成說明：2026-03-13）
