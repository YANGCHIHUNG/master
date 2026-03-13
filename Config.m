classdef Config
    %CONFIG Centralized constants for 5G RAN slicing simulation.
    %   This class defines all physical-layer, traffic, QoS, and RL
    %   constants used across the project to avoid hardcoded magic numbers.
    %   Units:
    %   - Bandwidth in Hz
    %   - Power in dBm
    %   - Delay in seconds
    %   - Data rates in bps
    %   - Queue sizes in bits
    %
    %   Access constants with:
    %       value = Config.<PropertyName>

    properties (Constant)
        % Physical layer constants
        W (1,1) double = 180e3
        B_RBs (1,1) double = 20
        M_mini_slots (1,1) double = 7
        Noise_Power (1,1) double = -114.0
        Tx_Power_dBm (1,1) double = 30.0
        Path_Loss_dB (1,1) double = 130.0
        Slot_duration (1,1) double = 0.5e-3
        Symbols_per_slot (1,1) double = 14
        Symbols_per_mini_slot (1,1) double = 2
        Subcarriers_per_RB (1,1) double = 12

        % Traffic and QoS parameters
        % lambda_urllc (1,1) double = 3.0
        lambda_urllc (1,1) double = 0.5
        D_one (1,1) double = 64
        tau_req (1,1) double = 0.5e-3
        lambda_embb (1,1) double = 0.001
        embb_xm_bits (1,1) double = 5e4
        embb_alpha (1,1) double = 1.2
        embb_penalty_scale (1,1) double = 1e-6

        % Simulation parameters
        Max_Episode_Steps (1,1) double = 700

        % RL and state abstraction hyperparameters
        N_g (1,1) double = 5
        Q_levels (1,1) double = 12
        phi_th (1,1) double = 10000.0
        omega_1 (1,1) double = 0.5
        omega_2 (1,1) double = 0.25
        omega_3 (1,1) double = 0.25
        sic_threshold (1,1) double = 0.5
        sic_min_superposition (1,1) double = 0.51

        % Observation normalization constants (avoid magic numbers in code)
        % - obs_urlllc_log_scale: divisor for log10-scaled URLLC queues
        % - obs_embb_log_scale : divisor for log10-scaled eMBB queues
        obs_urlllc_log_scale (1,1) double = 6.0
        obs_embb_log_scale (1,1) double = 9.0

        % Reward shaping / scaling constants (make magic numbers configurable)
        % - reward_fairness_weight: multiplier applied to Jain fairness term
        % - reward_scale          : final divisor to scale reward magnitude
        reward_fairness_weight (1,1) double = 0.1
        reward_scale (1,1) double = 100.0

        % Training script hyperparameters and execution settings
        random_seed (1,1) double = 42
        sac_minibatch_size (1,1) double = 256
        sac_actor_learn_rate (1,1) double = 1e-4
        sac_critic_learn_rate (1,1) double = 1e-3
        train_max_episodes (1,1) double = 2000
        train_save_every_episodes (1,1) double = 500
        train_save_dir (1,1) string = "models"

        % 5G Toolbox phase-1 PHY upgrade settings
        phy_enable_5g_toolbox (1,1) logical = true
        phy_mode (1,1) string = "tdl_tbs"

        % Carrier and numerology configuration for FR1 study grid.
        nr_frequency_range (1,1) string = "FR1"
        nr_center_frequency_hz (1,1) double = 3.5e9
        nr_numerology_mu (1,1) double = 1
        nr_subcarrier_spacing_khz (1,1) double = 30.0
        nr_cyclic_prefix (1,1) string = "normal"
        nr_n_cell_id (1,1) double = 1
        nr_n_size_grid (1,1) double = 20
        nr_n_start_grid (1,1) double = 0

        % 3GPP TR 38.901 path-loss and fading configuration.
        channel_model_type (1,1) string = "TDL"
        channel_delay_profile (1,1) string = "TDL-C"
        channel_delay_spread_s (1,1) double = 300e-9
        channel_scenario (1,1) string = "UMa"
        channel_is_los (1,1) logical = false
        channel_bs_position_m (3,1) double = [0.0; 0.0; 25.0]
        channel_ue_urlcc_position_m (3,1) double = [80.0; 20.0; 1.5]
        channel_ue_embb_position_m (3,1) double = [120.0; -40.0; 1.5]
        channel_use_pathloss (1,1) logical = true
        channel_use_shadow_fading (1,1) logical = true
        channel_normalize_path_gains (1,1) logical = true
        channel_normalize_outputs (1,1) logical = true
        channel_random_stream (1,1) string = "mt19937ar with seed"
        channel_seed_base (1,1) double = 73

        % Mobility and Doppler settings.
        channel_ue_speed_kmh (1,1) double = 30.0
        channel_max_doppler_hz (1,1) double = 97.28952776612769

        % Link adaptation and transport-block settings.
        la_enable_csi_report (1,1) logical = true
        la_cqi_table (1,1) string = "table1"
        la_num_layers (1,1) double = 1
        la_modulation_default (1,1) string = "64QAM"
        la_mcs_table (1,1) string = "qam64"
        la_x_overhead (1,1) double = 0
        la_tb_scaling (1,1) double = 1.0
        la_pdsch_mapping_type (1,1) string = "A"
        la_pdsch_symbol_allocation (1,2) double = [0, 14]
        la_target_bler (1,1) double = 0.1
        la_use_wideband_cqi (1,1) logical = true
        la_cqi_smoothing_alpha (1,1) double = 0.7

        % Observation/logging toggles for future phases.
        obs_include_phy_cqi (1,1) logical = true
        obs_include_phy_tbs (1,1) logical = false
        obs_include_pathloss (1,1) logical = false
    end
end
