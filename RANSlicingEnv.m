classdef RANSlicingEnv < rl.env.MATLABEnvironment
    %RANSLICINGENV Custom RL environment for 5G RAN slicing.
    %   This environment models one eMBB slot composed of
    %   Config.M_mini_slots mini-slots. The observation is a column vector:
    %   [q_u_G1..q_u_G5, q_e_G1..q_e_G5, m]^T with size [11 x 1].
    %
    %   Observation and transition logic follow the project MDP
    %   specification for mini-slot level control.

    properties
        % Queue states in bits for grouped traffic, size [Config.N_g x 1].
        URLLCGroupQueues
        eMBBGroupQueues

        % Channel power gains |h|^2 per RB, size [Config.B_RBs x 1].
        URLLCChannelGain
        eMBBChannelGain

        % eMBB QoS demand per RB in bps, size [Config.B_RBs x 1].
        eMBBQoS

        % Current mini-slot index in [0, Config.M_mini_slots-1].
        MiniSlotIndex (1,1) double = 0

        % Historical eMBB rates used by reward/stability, reset each episode.
        eMBBRateHistory
    end

    properties (Access = private)
        % Latest observation column vector, size [11 x 1].
        Observation
        TraceURLLC
        TraceeMBB
        GlobalStepIndex
        MaxTraceSteps
    end

    methods
        function this = RANSlicingEnv()
            %RANSLICINGENV Construct environment with action/observation specs.
            %   Action:
            %   - [Config.N_g x 1], continuous in [0, 1], representing
            %     group-wise power allocation ratio theta_Gk.
            %
            %   Observation:
            %   - [2*Config.N_g + 1 x 1], bounded in [0, Config.Q_levels-1]
            %     for queue quantization levels and mini-slot index m.

            actInfo = rlNumericSpec([Config.N_g 1], ...
                "LowerLimit", 0, ...
                "UpperLimit", 1);
            actInfo.Name = "group_power_ratio";
            actInfo.Description = "theta for each channel-quality group";

            obsDimension = 2 * Config.N_g + 1;
            obsInfo = rlNumericSpec([obsDimension 1], ...
                "LowerLimit", 0, ...
                "UpperLimit", Config.Q_levels - 1);
            obsInfo.Name = "queue_and_time_state";
            obsInfo.Description = "quantized URLLC/eMBB queues and mini-slot index";

            this = this@rl.env.MATLABEnvironment(obsInfo, actInfo);

            traceData = load("ChannelTraces.mat");
            this.TraceURLLC = traceData.urllcChannelTrace;
            this.TraceeMBB = traceData.embbChannelTrace;
            this.MaxTraceSteps = size(this.TraceURLLC, 2);
            this.GlobalStepIndex = randi([1, this.MaxTraceSteps]);

            % Initialize state for a fresh episode.
            this.Observation = reset(this);
        end

        function initialObservation = reset(this)
            %RESET Start a new eMBB slot episode (1 ms).
            %   initialObservation = reset(this)
            %
            %   Output:
            %   - initialObservation: [11 x 1] column vector
            %     [q_u_G1..q_u_G5, q_e_G1..q_e_G5, m]^T.
            %
            %   Reset behavior:
            %   1) Read channel gains from offline trace by circular index.
            %   2) Draw random eMBB QoS uniformly in
            %      [Config.eMBB_QoS_min, Config.eMBB_QoS_max].
            %   3) Set grouped queues to zero and m = 0.
            %   4) Clear rate history buffer.

            rbCount = Config.B_RBs;

            safeIndex = mod(this.GlobalStepIndex - 1, this.MaxTraceSteps) + 1;
            this.URLLCChannelGain = this.TraceURLLC(:, safeIndex);
            this.eMBBChannelGain = this.TraceeMBB(:, safeIndex);

            qosSpan = Config.eMBB_QoS_max - Config.eMBB_QoS_min;
            this.eMBBQoS = Config.eMBB_QoS_min + qosSpan .* rand(rbCount, 1);

            this.URLLCGroupQueues = zeros(Config.N_g, 1);
            this.eMBBGroupQueues = zeros(Config.N_g, 1);
            this.MiniSlotIndex = 0;
            this.eMBBRateHistory = zeros(0, 1);

            queueState = [
                ChannelStateProcessor.quantizeQueue(this.URLLCGroupQueues);
                ChannelStateProcessor.quantizeQueue(this.eMBBGroupQueues)
            ];
            initialObservation = [queueState; this.MiniSlotIndex];
            this.Observation = initialObservation;
        end

        function [nextObservation, reward, isDone, loggedSignals] = step(this, action)
            %STEP Apply group-wise power allocation and advance one mini-slot.
            %   [nextObservation, reward, isDone, loggedSignals] = step(this, action)
            %
            %   Input:
            %   - action: [Config.N_g x 1] vector of group-wise power ratios
            %     theta_Gk in [0, 1].
            %
            %   Outputs:
            %   - nextObservation: [11 x 1] quantized queue and mini-slot state.
            %   - reward: Scalar reward from calculateReward().
            %   - isDone: Logical terminal flag, true when m == Config.M_mini_slots.
            %   - loggedSignals: Struct with:
            %       urllc_actual_delay, embb_satisfaction, embb_fairness,
            %       is_urllc_failed.
            %
            %   NOMA/SIC clamping policy:
            %   - If |h_u|^2 > |h_e|^2:
            %       theta_raw >= 0.5 -> theta_b = 1.0
            %       theta_raw <  0.5 -> theta_b = 0.0
            %   - If |h_u|^2 <= |h_e|^2:
            %       theta_raw > 0    -> theta_b = max(0.51, theta_raw)
            %       theta_raw == 0   -> theta_b = 0.0

            actionVec = min(1.0, max(0.0, double(action(:))));
            if numel(actionVec) ~= Config.N_g
                error("RANSlicingEnv:InvalidActionSize", ...
                    "Action must contain exactly %d elements.", Config.N_g);
            end

            rbCount = Config.B_RBs;
            slotDuration = Config.Slot_duration;
            miniSlotDuration = slotDuration / Config.M_mini_slots;
            noisePowerLinear = 10 ^ (Config.Noise_Power / 10);

            % Map each RB to a channel-quality group using eMBB link quality.
            embbSNRLinear = this.eMBBChannelGain ./ max(noisePowerLinear, eps);
            embbSNRdB = 10 * log10(max(eps, embbSNRLinear));
            mcsPerRB = ChannelStateProcessor.snrToMCS(embbSNRdB);
            groupMembers = ChannelStateProcessor.groupUsers(mcsPerRB);

            rbToGroup = ones(rbCount, 1) * Config.N_g;
            for groupId = 1:Config.N_g
                rbToGroup(groupMembers{groupId}) = groupId;
            end

            % eMBB offered load arrival per mini-slot from QoS targets.
            for groupId = 1:Config.N_g
                rbIdx = groupMembers{groupId};
                if ~isempty(rbIdx)
                    demandBits = sum(this.eMBBQoS(rbIdx)) * miniSlotDuration;
                    this.eMBBGroupQueues(groupId) = this.eMBBGroupQueues(groupId) + demandBits;
                end
            end

            initialURLLCBits = sum(this.URLLCGroupQueues);
            remainingURLLCBits = initialURLLCBits;
            embbRatesPerRB = zeros(rbCount, 1);

            [~, sortedRBs] = sort(this.URLLCChannelGain, "descend");
            for idx = 1:rbCount
                rb = sortedRBs(idx);
                groupId = rbToGroup(rb);
                thetaRaw = actionVec(groupId);
                hub = this.URLLCChannelGain(rb);
                heb = this.eMBBChannelGain(rb);

                if hub > heb
                    if thetaRaw >= Config.sic_threshold
                        thetaB = 1.0;
                    else
                        thetaB = 0.0;
                    end
                else
                    if thetaRaw > 0
                        thetaB = max(Config.sic_min_superposition, thetaRaw);
                    else
                        thetaB = 0.0;
                    end
                end

                if thetaB > 0
                    urllcSINR = (thetaB * hub) / (noisePowerLinear + (1 - thetaB) * hub + eps);
                    urllcRate = Config.W * log2(1 + urllcSINR);
                    urllcBits = urllcRate * miniSlotDuration;

                    servedURLLCBits = min(urllcBits, remainingURLLCBits);
                    remainingURLLCBits = remainingURLLCBits - servedURLLCBits;

                    % Deduct served URLLC bits from group queues in index order.
                    bitsToDeduct = servedURLLCBits;
                    for qGroupId = 1:Config.N_g
                        if bitsToDeduct <= 0
                            break;
                        end
                        drained = min(this.URLLCGroupQueues(qGroupId), bitsToDeduct);
                        this.URLLCGroupQueues(qGroupId) = this.URLLCGroupQueues(qGroupId) - drained;
                        bitsToDeduct = bitsToDeduct - drained;
                    end
                end

                if thetaB > 0
                    embbSINR = ((1 - thetaB) * heb) / (noisePowerLinear + thetaB * heb + eps);
                else
                    embbSINR = heb / (noisePowerLinear + eps);
                end
                embbRatesPerRB(rb) = Config.W * log2(1 + embbSINR);

                embbBits = embbRatesPerRB(rb) * miniSlotDuration;
                drainedEMBB = min(this.eMBBGroupQueues(groupId), embbBits);
                this.eMBBGroupQueues(groupId) = this.eMBBGroupQueues(groupId) - drainedEMBB;

                if remainingURLLCBits <= 0
                    break;
                end
            end

            this.URLLCGroupQueues = max(0.0, this.URLLCGroupQueues);
            this.eMBBGroupQueues = max(0.0, this.eMBBGroupQueues);

            if initialURLLCBits <= 0
                % No URLLC traffic this step
                actualDelay = 0.0;
            elseif remainingURLLCBits <= eps
                % Agent successfully cleared the queue within this 1 mini-slot
                actualDelay = miniSlotDuration;
            else
                % Agent failed to clear the queue. Trigger soft penalty by exceeding tau_req
                untransmittedRatio = remainingURLLCBits / max(initialURLLCBits, eps);
                actualDelay = Config.tau_req + (untransmittedRatio * miniSlotDuration);
            end

            reward = calculateReward(actualDelay, Config.tau_req, embbRatesPerRB, this.eMBBQoS);

            satisfactionRatios = min(1.0, embbRatesPerRB ./ max(this.eMBBQoS, eps));
            embbSatisfaction = mean(satisfactionRatios);
            if all(embbRatesPerRB == 0)
                embbFairness = 0.0;
            else
                fairnessDenominator = numel(satisfactionRatios) * sum(satisfactionRatios .^ 2);
                embbFairness = (sum(satisfactionRatios) ^ 2) / max(fairnessDenominator, eps);
            end

            isURLLCFailed = actualDelay > Config.tau_req;
            loggedSignals = struct( ...
                "urllc_actual_delay", actualDelay, ...
                "embb_satisfaction", embbSatisfaction, ...
                "embb_fairness", embbFairness, ...
                "is_urllc_failed", isURLLCFailed);

            newURLLCPackets = this.samplePoisson(Config.lambda_urllc, Config.N_g, 1);
            this.URLLCGroupQueues = this.URLLCGroupQueues + newURLLCPackets * Config.D_one;

            this.MiniSlotIndex = this.MiniSlotIndex + 1;
            this.GlobalStepIndex = this.GlobalStepIndex + 1;
            isDone = this.MiniSlotIndex >= Config.M_mini_slots;

            if ~isDone
                safeIndex = mod(this.GlobalStepIndex - 1, this.MaxTraceSteps) + 1;
                this.URLLCChannelGain = this.TraceURLLC(:, safeIndex);
                this.eMBBChannelGain = this.TraceeMBB(:, safeIndex);
            end

            this.eMBBRateHistory = [this.eMBBRateHistory; mean(embbRatesPerRB)];

            queueState = [
                ChannelStateProcessor.quantizeQueue(this.URLLCGroupQueues);
                ChannelStateProcessor.quantizeQueue(this.eMBBGroupQueues)
            ];
            nextObservation = [queueState; this.MiniSlotIndex];
            this.Observation = nextObservation;

            notifyEnvUpdated(this);
        end
    end

    methods (Static, Access = private)
        function samples = samplePoisson(lambda, rows, cols)
            %SAMPLEPOISSON Draw Poisson samples with toolbox fallback.
            %   samples = samplePoisson(lambda, rows, cols) returns a
            %   [rows x cols] matrix. Uses poissrnd if available, otherwise
            %   uses Knuth's algorithm for scalar lambda.

            if exist("poissrnd", "file") == 2
                samples = poissrnd(lambda, rows, cols);
                return;
            end

            if ~isscalar(lambda) || lambda < 0
                error("RANSlicingEnv:InvalidPoissonLambda", ...
                    "Fallback Poisson sampler requires nonnegative scalar lambda.");
            end

            samples = zeros(rows, cols);
            threshold = exp(-lambda);
            for sampleIdx = 1:numel(samples)
                k = 0;
                p = 1.0;
                while p > threshold
                    k = k + 1;
                    p = p * rand();
                end
                samples(sampleIdx) = k - 1;
            end
        end
    end
end
