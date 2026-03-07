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
        Slot_duration (1,1) double = 1e-3

        % Traffic and QoS parameters
        lambda_urllc (1,1) double = 3.0
        D_one (1,1) double = 256
        tau_req (1,1) double = 0.5e-3
        eMBB_QoS_min (1,1) double = 20e6
        eMBB_QoS_max (1,1) double = 50e6

        % RL and state abstraction hyperparameters
        N_g (1,1) double = 5
        Q_levels (1,1) double = 12
        phi_th (1,1) double = 10000.0
        omega_1 (1,1) double = 0.5
        omega_2 (1,1) double = 0.25
        omega_3 (1,1) double = 0.25
        sic_threshold (1,1) double = 0.5
        sic_min_superposition (1,1) double = 0.51

        % Training script hyperparameters and execution settings
        random_seed (1,1) double = 42
        sac_minibatch_size (1,1) double = 256
        sac_actor_learn_rate (1,1) double = 1e-4
        sac_critic_learn_rate (1,1) double = 1e-3
        train_max_episodes (1,1) double = 2000
        train_save_every_episodes (1,1) double = 500
        train_save_dir (1,1) string = "models"
    end
end
