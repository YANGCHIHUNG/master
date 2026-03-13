classdef RANSlicingEnv < rl.env.MATLABEnvironment
    %RANSLICINGENV Custom RL environment for 5G RAN slicing.
    %   This environment models one NR slot composed of
    %   Config.M_mini_slots mini-slots. The observation is a column vector:
    %   [u_G1..u_G5, e_G1..e_G5, m, cqi_u, cqi_e]^T with size [13 x 1],
    %   where each component is normalized to a small numeric range for
    %   RL training.
    %
    %   Phase-1 PHY integration replaces trace replay and Shannon-style
    %   rates with:
    %   1) 5G Toolbox TDL-C fading and scenario-based path loss.
    %   2) Wideband CQI estimation per PRB.
    %   3) CQI-driven MCS/TBS service for URLLC puncturing and eMBB.

    properties
        % Queue states in bits for grouped traffic, size [Config.N_g x 1].
        URLLCGroupQueues
        eMBBGroupQueues

        % Runtime-overridable eMBB arrival rate for load stress tests.
        lambda_embb (1,1) double = Config.lambda_embb

        % Effective post-pathloss signal power per RB, size [Config.B_RBs x 1].
        URLLCChannelGain
        eMBBChannelGain

        % Latest PRB-wise CQI estimates for diagnostics and grouping.
        URLLCPRBCQI
        eMBBPRBCQI

        % Latest wideband CQI summaries from NRPhyChannel.
        URLLCWidebandCQI (1,1) double = 0
        eMBBWidebandCQI (1,1) double = 0

        % Current cyclic mini-slot index in [1, Config.M_mini_slots].
        MiniSlotIndex (1,1) double = 1

        % Historical eMBB rates used by reward/stability, reset each episode.
        eMBBRateHistory
    end

    properties (Access = private)
        % Latest observation column vector, size [13 x 1].
        Observation
        PhyChannel
        GlobalStepIndex
        CurrentStep
        URLLCPacketBits
        URLLCPacketArrivalSteps
    end

    methods
        function this = RANSlicingEnv()
            %RANSLICINGENV Construct environment with action/observation specs.
            %   Action:
            %   - [Config.N_g x 1], continuous in [0, 1], representing
            %     group-wise URLLC puncturing quota theta_Gk.
            %
            %   Observation:
            %   - [2*Config.N_g + 3 x 1], bounded in [0, 1], containing
            %     normalized URLLC queues, normalized eMBB queues,
            %     normalized mini-slot index, normalized URLLC CQI, and
            %     normalized eMBB CQI.

            actInfo = rlNumericSpec([Config.N_g 1], ...
                "LowerLimit", 0, ...
                "UpperLimit", 1);
            actInfo.Name = "group_prb_quota";
            actInfo.Description = "theta for group-wise URLLC puncturing quota";

            obsDimension = 2 * Config.N_g + 3;
            obsInfo = rlNumericSpec([obsDimension 1], ...
                "LowerLimit", 0, ...
                "UpperLimit", 1);
            obsInfo.Name = "queue_and_time_state";
            obsInfo.Description = ...
                "normalized URLLC/eMBB queues, mini-slot index, URLLC CQI, and eMBB CQI";

            this = this@rl.env.MATLABEnvironment(obsInfo, actInfo);
            this.PhyChannel = NRPhyChannel();
            this.GlobalStepIndex = 0;

            % Initialize state for a fresh episode.
            this.Observation = reset(this);
        end

        function initialObservation = reset(this)
            %RESET Start a new NR slot episode using a fresh PHY realization.
            %   initialObservation = reset(this)
            %
            %   Output:
            %   - initialObservation: [13 x 1] normalized column vector
            %     [u_G1..u_G5, e_G1..e_G5, m, cqi_u, cqi_e]^T.

            this.URLLCGroupQueues = zeros(Config.N_g, 1);
            this.eMBBGroupQueues = zeros(Config.N_g, 1);
            this.MiniSlotIndex = 1;
            this.CurrentStep = 0;
            this.GlobalStepIndex = 0;
            this.eMBBRateHistory = zeros(0, 1);
            this.URLLCPacketBits = cell(Config.N_g, 1);
            this.URLLCPacketArrivalSteps = cell(Config.N_g, 1);

            this.PhyChannel.resetEpisode(randi([0, 2^31 - 1]));
            initialMetrics = this.PhyChannel.stepChannel( ...
                1, ...
                this.MiniSlotIndex, ...
                zeros(Config.N_g, 1), ...
                false);
            this.applyPhyMetrics(initialMetrics);

            initialObservation = this.buildObservation();
            this.Observation = initialObservation;
        end

        function [nextObservation, reward, isDone, loggedSignals] = step(this, action)
            %STEP Apply group-wise PRB puncturing quota and advance one mini-slot.
            %   [nextObservation, reward, isDone, loggedSignals] = step(this, action)
            %
            %   Input:
            %   - action: [Config.N_g x 1] vector of group-wise URLLC PRB
            %     puncturing quotas theta_Gk in [0, 1].
            %
            %   Outputs:
            %   - nextObservation: [13 x 1] normalized queue, time, and CQI state.
            %   - reward: Scalar reward from calculateReward().
            %   - isDone: Logical terminal flag, true when CurrentStep reaches
            %     Config.Max_Episode_Steps.
            %   - loggedSignals: Struct with:
            %       urllc_actual_delay, embb_satisfaction, embb_fairness,
            %       is_urllc_failed, and PHY diagnostics.

            actionVec = min(1.0, max(0.0, double(action(:))));
            if any(isnan(actionVec)) || any(isinf(actionVec))
                fprintf("CRITICAL ERROR: Action contains NaN or Inf at Global Step %d!\n", ...
                    this.GlobalStepIndex);
                disp(actionVec');
            end
            if numel(actionVec) ~= Config.N_g
                error("RANSlicingEnv:InvalidActionSize", ...
                    "Action must contain exactly %d elements.", Config.N_g);
            end

            this.CurrentStep = this.CurrentStep + 1;
            this.GlobalStepIndex = this.GlobalStepIndex + 1;

            miniSlotDuration = Config.Slot_duration / Config.M_mini_slots;
            currentServiceStep = this.CurrentStep;

            this.ensureURLLCPacketState();

            % 3GPP FTP3 bursty traffic arrivals for eMBB groups.
            for groupId = 1:Config.N_g
                if rand() < this.lambda_embb
                    fileSizeBits = this.sampleParetoBits( ...
                        Config.embb_xm_bits, ...
                        Config.embb_alpha);
                    this.eMBBGroupQueues(groupId) = ...
                        this.eMBBGroupQueues(groupId) + fileSizeBits;
                end
            end

            initialURLLCBits = sum(this.URLLCGroupQueues);
            phyMetrics = this.PhyChannel.stepChannel( ...
                this.CurrentStep, ...
                this.MiniSlotIndex, ...
                actionVec, ...
                initialURLLCBits > eps);
            this.applyPhyMetrics(phyMetrics);

            embbGroupServedBits = zeros(Config.N_g, 1);
            maxURLLCDelay = 0.0;

            requestedURLLCBits = sum(phyMetrics.groupURLLCTBSBits);
            if requestedURLLCBits > eps
                [~, servedDelay] = this.serveURLLCBits( ...
                    requestedURLLCBits, ...
                    currentServiceStep, ...
                    miniSlotDuration);
                maxURLLCDelay = max(maxURLLCDelay, servedDelay);
            end

            for groupId = 1:Config.N_g
                drainedEMBB = min( ...
                    this.eMBBGroupQueues(groupId), ...
                    phyMetrics.groupEMBBTBSBits(groupId));
                this.eMBBGroupQueues(groupId) = this.eMBBGroupQueues(groupId) - drainedEMBB;
                embbGroupServedBits(groupId) = embbGroupServedBits(groupId) + drainedEMBB;
            end

            this.URLLCGroupQueues = max(0.0, this.URLLCGroupQueues);
            this.eMBBGroupQueues = max(0.0, this.eMBBGroupQueues);
            embbGroupActualRates = embbGroupServedBits / miniSlotDuration;

            if initialURLLCBits <= 0
                actualDelay = 0.0;
            else
                maxURLLCDelay = max(maxURLLCDelay, ...
                    this.getOldestURLLCDelay(currentServiceStep, miniSlotDuration));
                actualDelay = maxURLLCDelay;
            end

            reward = calculateReward( ...
                actualDelay, ...
                Config.tau_req, ...
                embbGroupActualRates, ...
                this.eMBBGroupQueues);

            if isnan(reward) || isinf(reward)
                fprintf("CRITICAL ERROR: Reward is %f at Global Step %d!\n", ...
                    reward, this.GlobalStepIndex);
            end
            if mod(this.CurrentStep, 200) == 0
                fprintf(['[Env Heartbeat] Ep Step: %d | URLLC Q: %.1f | ' ...
                    'eMBB Q: %.1f | Reward: %.4f | CQI(U/E): %d/%d\n'], ...
                    this.CurrentStep, ...
                    sum(this.URLLCGroupQueues), ...
                    sum(this.eMBBGroupQueues), ...
                    reward, ...
                    this.URLLCWidebandCQI, ...
                    this.eMBBWidebandCQI);
            end

            servedEmbbBits = sum(embbGroupServedBits);
            totalEmbbQueue = sum(this.eMBBGroupQueues) + servedEmbbBits;
            if totalEmbbQueue > 0
                embbSatisfaction = servedEmbbBits / totalEmbbQueue;
            else
                embbSatisfaction = 1.0;
            end

            activeEmbbGroupIdx = this.eMBBGroupQueues > 0;
            activeEmbbRates = embbGroupActualRates(activeEmbbGroupIdx);
            if isempty(activeEmbbRates) || sum(activeEmbbRates) == 0
                embbFairness = 1.0;
            else
                fairnessDenominator = numel(activeEmbbRates) * sum(activeEmbbRates .^ 2);
                embbFairness = (sum(activeEmbbRates) ^ 2) / max(fairnessDenominator, eps);
            end

            isURLLCFailed = actualDelay > Config.tau_req;
            loggedSignals = struct( ...
                "urllc_actual_delay", actualDelay, ...
                "embb_satisfaction", embbSatisfaction, ...
                "embb_fairness", embbFairness, ...
                "is_urllc_failed", isURLLCFailed, ...
                "urllc_wideband_cqi", this.URLLCWidebandCQI, ...
                "embb_wideband_cqi", this.eMBBWidebandCQI, ...
                "urllc_tbs_bits", sum(phyMetrics.groupURLLCTBSBits), ...
                "embb_tbs_bits", sum(phyMetrics.groupEMBBTBSBits));

            newURLLCPackets = this.samplePoisson(Config.lambda_urllc, Config.N_g, 1);
            this.addURLLCPackets(newURLLCPackets, currentServiceStep + 1);

            isDone = this.CurrentStep >= Config.Max_Episode_Steps;
            this.MiniSlotIndex = mod(this.CurrentStep, Config.M_mini_slots) + 1;
            this.eMBBRateHistory = [this.eMBBRateHistory; mean(embbGroupActualRates)];

            nextObservation = this.buildObservation();
            this.Observation = nextObservation;

            notifyEnvUpdated(this);
        end
    end

    methods (Access = private)
        function applyPhyMetrics(this, phyMetrics)
            this.URLLCChannelGain = phyMetrics.urllcSignalPowerPerRB(:);
            this.eMBBChannelGain = phyMetrics.embbSignalPowerPerRB(:);
            this.URLLCPRBCQI = phyMetrics.urllcCQIPerRB(:);
            this.eMBBPRBCQI = phyMetrics.embbCQIPerRB(:);
            this.URLLCWidebandCQI = phyMetrics.urllcWidebandCQI;
            this.eMBBWidebandCQI = phyMetrics.embbWidebandCQI;
        end

        function observation = buildObservation(this)
            % Use configurable normalization constants from Config to avoid
            % hard-coded magic numbers and allow easy tuning.
            normalizedURLLCQueues = log10(1.0 + this.URLLCGroupQueues) / Config.obs_urlllc_log_scale;
            normalizedEMBBQueues = log10(1.0 + this.eMBBGroupQueues) / Config.obs_embb_log_scale;
            normalizedURLLCQueues = min(1.0, max(0.0, normalizedURLLCQueues));
            normalizedEMBBQueues = min(1.0, max(0.0, normalizedEMBBQueues));
            normalizedMiniSlotIndex = min(1.0, max(0.0, ...
                this.MiniSlotIndex / Config.M_mini_slots));
            normalizedURLLCCQI = min(1.0, max(0.0, this.URLLCWidebandCQI / 15.0));
            normalizedEMBBCQI = min(1.0, max(0.0, this.eMBBWidebandCQI / 15.0));

            observation = [
                normalizedURLLCQueues
                normalizedEMBBQueues
                normalizedMiniSlotIndex
                normalizedURLLCCQI
                normalizedEMBBCQI
            ];
        end

        function ensureURLLCPacketState(this)
            if isempty(this.URLLCPacketBits) || numel(this.URLLCPacketBits) ~= Config.N_g
                this.URLLCPacketBits = cell(Config.N_g, 1);
                this.URLLCPacketArrivalSteps = cell(Config.N_g, 1);
            end

            bufferedBits = this.computeURLLCQueueBits();
            if any(abs(bufferedBits - this.URLLCGroupQueues) > eps)
                this.rebuildURLLCBuffersFromQueues(this.CurrentStep);
            end
        end

        function [servedBits, maxServedDelay] = serveURLLCBits(this, requestedBits, serviceStep, miniSlotDuration)
            servedBits = 0.0;
            maxServedDelay = 0.0;
            remainingRequest = max(0.0, requestedBits);

            for groupId = 1:Config.N_g
                if remainingRequest <= eps
                    break;
                end

                packetBits = this.URLLCPacketBits{groupId};
                arrivalSteps = this.URLLCPacketArrivalSteps{groupId};
                packetIdx = 1;
                while packetIdx <= numel(packetBits) && remainingRequest > eps
                    drained = min(packetBits(packetIdx), remainingRequest);
                    packetBits(packetIdx) = packetBits(packetIdx) - drained;
                    remainingRequest = remainingRequest - drained;
                    servedBits = servedBits + drained;

                    if packetBits(packetIdx) <= eps
                        packetDelay = (serviceStep - arrivalSteps(packetIdx) + 1) * miniSlotDuration;
                        maxServedDelay = max(maxServedDelay, packetDelay);
                        packetBits(packetIdx) = [];
                        arrivalSteps(packetIdx) = [];
                    else
                        packetIdx = packetIdx + 1;
                    end
                end

                this.URLLCPacketBits{groupId} = packetBits;
                this.URLLCPacketArrivalSteps{groupId} = arrivalSteps;
            end

            this.URLLCGroupQueues = this.computeURLLCQueueBits();
        end

        function addURLLCPackets(this, packetCounts, arrivalStep)
            this.ensureURLLCPacketState();

            for groupId = 1:Config.N_g
                packetCount = max(0, round(double(packetCounts(groupId))));
                if packetCount <= 0
                    continue;
                end

                this.URLLCPacketBits{groupId} = [
                    this.URLLCPacketBits{groupId}
                    repmat(Config.D_one, packetCount, 1)
                ];
                this.URLLCPacketArrivalSteps{groupId} = [
                    this.URLLCPacketArrivalSteps{groupId}
                    repmat(arrivalStep, packetCount, 1)
                ];
            end

            this.URLLCGroupQueues = this.computeURLLCQueueBits();
        end

        function delay = getOldestURLLCDelay(this, serviceStep, miniSlotDuration)
            oldestArrival = inf;
            for groupId = 1:Config.N_g
                arrivalSteps = this.URLLCPacketArrivalSteps{groupId};
                if ~isempty(arrivalSteps)
                    oldestArrival = min(oldestArrival, arrivalSteps(1));
                end
            end

            if isinf(oldestArrival)
                delay = 0.0;
            else
                delay = (serviceStep - oldestArrival + 1) * miniSlotDuration;
            end
        end

        function queueBits = computeURLLCQueueBits(this)
            queueBits = zeros(Config.N_g, 1);
            if isempty(this.URLLCPacketBits)
                return;
            end

            for groupId = 1:Config.N_g
                queueBits(groupId) = sum(this.URLLCPacketBits{groupId});
            end
        end

        function rebuildURLLCBuffersFromQueues(this, arrivalStep)
            this.URLLCPacketBits = cell(Config.N_g, 1);
            this.URLLCPacketArrivalSteps = cell(Config.N_g, 1);

            for groupId = 1:Config.N_g
                queueBits = max(0.0, double(this.URLLCGroupQueues(groupId)));
                wholePackets = floor(queueBits / Config.D_one);
                residualBits = queueBits - wholePackets * Config.D_one;
                packetBits = repmat(Config.D_one, wholePackets, 1);
                if residualBits > eps
                    packetBits = [packetBits; residualBits];
                end

                this.URLLCPacketBits{groupId} = packetBits;
                this.URLLCPacketArrivalSteps{groupId} = repmat(arrivalStep, numel(packetBits), 1);
            end

            this.URLLCGroupQueues = this.computeURLLCQueueBits();
        end
    end

    methods (Static, Access = private)
        function sample = sampleParetoBits(xm, alpha)
            if xm <= 0 || alpha <= 0
                error("RANSlicingEnv:InvalidParetoParameters", ...
                    "Pareto parameters xm and alpha must be positive.");
            end

            uniformSample = max(rand(), eps);
            sample = xm / (uniformSample ^ (1 / alpha));
            sample = min(sample, 200e6);
        end

        function samples = samplePoisson(lambda, rows, cols)
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
