classdef RANSlicingEnv < rl.env.MATLABEnvironment
    %RANSLICINGENV Custom RL environment for 5G RAN slicing.
    %   This environment models one eMBB slot composed of
    %   Config.M_mini_slots mini-slots. The observation is a column vector:
    %   [u_G1..u_G5, e_G1..e_G5, m]^T with size [11 x 1], where each
    %   component is normalized to a small numeric range for RL training.
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

        % Current cyclic mini-slot index in [1, Config.M_mini_slots].
        MiniSlotIndex (1,1) double = 1

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
        CurrentStep
        URLLCPacketBits
        URLLCPacketArrivalSteps
    end

    methods
        function this = RANSlicingEnv()
            %RANSLICINGENV Construct environment with action/observation specs.
            %   Action:
            %   - [Config.N_g x 1], continuous in [0, 1], representing
            %     group-wise power allocation ratio theta_Gk.
            %
            %   Observation:
            %   - [2*Config.N_g + 1 x 1], bounded in [0, 1], containing
            %     normalized URLLC queues, normalized eMBB queues, and
            %     normalized mini-slot index.

            actInfo = rlNumericSpec([Config.N_g 1], ...
                "LowerLimit", 0, ...
                "UpperLimit", 1);
            actInfo.Name = "group_power_ratio";
            actInfo.Description = "theta for each channel-quality group";

            obsDimension = 2 * Config.N_g + 1;
            obsInfo = rlNumericSpec([obsDimension 1], ...
                "LowerLimit", 0, ...
                "UpperLimit", 1);
            obsInfo.Name = "queue_and_time_state";
            obsInfo.Description = "normalized URLLC/eMBB queues and mini-slot index";

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
            %   - initialObservation: [11 x 1] normalized column vector
            %     [u_G1..u_G5, e_G1..e_G5, m]^T.
            %
            %   Reset behavior:
            %   1) Read channel gains from offline trace by circular index.
            %   2) Set grouped queues to zero and m = 1.
            %   3) Clear rate history buffer.

            safeIndex = mod(this.GlobalStepIndex - 1, this.MaxTraceSteps) + 1;
            this.URLLCChannelGain = this.TraceURLLC(:, safeIndex);
            this.eMBBChannelGain = this.TraceeMBB(:, safeIndex);

            this.URLLCGroupQueues = zeros(Config.N_g, 1);
            this.eMBBGroupQueues = zeros(Config.N_g, 1);
            this.MiniSlotIndex = 1;
            this.CurrentStep = 0;
            this.eMBBRateHistory = zeros(0, 1);
            this.URLLCPacketBits = cell(Config.N_g, 1);
            this.URLLCPacketArrivalSteps = cell(Config.N_g, 1);

            initialObservation = this.buildObservation();
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
            %   - nextObservation: [11 x 1] normalized queue and mini-slot state.
            %   - reward: Scalar reward from calculateReward().
            %   - isDone: Logical terminal flag, true when CurrentStep reaches
            %     Config.Max_Episode_Steps.
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
            if any(isnan(actionVec)) || any(isinf(actionVec))
                fprintf('CRITICAL ERROR: Action contains NaN or Inf at Global Step %d!\n', ...
                    this.GlobalStepIndex);
                disp(actionVec');
            end
            if numel(actionVec) ~= Config.N_g
                error("RANSlicingEnv:InvalidActionSize", ...
                    "Action must contain exactly %d elements.", Config.N_g);
            end

            this.CurrentStep = this.CurrentStep + 1;

            rbCount = Config.B_RBs;
            slotDuration = Config.Slot_duration;
            miniSlotDuration = slotDuration / Config.M_mini_slots;
            noisePowerLinear = 10 ^ (Config.Noise_Power / 10);
            rxPowerFactor = 10 ^ ((Config.Tx_Power_dBm - Config.Path_Loss_dB) / 10);
            currentServiceStep = this.CurrentStep;

            this.ensureURLLCPacketState();

            % 3GPP FTP3 bursty traffic arrivals for eMBB groups.
            for groupId = 1:Config.N_g
                if rand() < Config.lambda_embb
                    fileSizeBits = this.sampleParetoBits( ...
                        Config.embb_xm_bits, ...
                        Config.embb_alpha);
                    this.eMBBGroupQueues(groupId) = ...
                        this.eMBBGroupQueues(groupId) + fileSizeBits;
                end
            end

            % Map each RB to a channel-quality group using eMBB CQI.
            embbSNRLinear = (this.eMBBChannelGain .* rxPowerFactor) ./ max(noisePowerLinear, eps);
            embbSNRdB = 10 * log10(max(eps, embbSNRLinear));
            cqiPerRB = ChannelStateProcessor.snrToCQI(embbSNRdB);
            groupMembers = ChannelStateProcessor.groupUsers(cqiPerRB);

            rbToGroup = ones(rbCount, 1) * Config.N_g;
            for groupId = 1:Config.N_g
                rbToGroup(groupMembers{groupId}) = groupId;
            end

            initialURLLCBits = sum(this.URLLCGroupQueues);
            remainingURLLCBits = initialURLLCBits;
            embbActualRates = zeros(rbCount, 1);
            maxURLLCDelay = 0.0;

            [~, sortedRBs] = sort(this.URLLCChannelGain, "descend");
            for idx = 1:rbCount
                rb = sortedRBs(idx);
                groupId = rbToGroup(rb);
                thetaRaw = actionVec(groupId);
                hub = this.URLLCChannelGain(rb) * rxPowerFactor;
                heb = this.eMBBChannelGain(rb) * rxPowerFactor;

                thetaB = 0.0;
                if remainingURLLCBits > eps
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
                end

                if thetaB > 0
                    urllcSINR = (thetaB * hub) / (noisePowerLinear + (1 - thetaB) * hub + eps);
                    urllcSINRdB = 10 * log10(max(eps, urllcSINR));
                    urllcCQI = ChannelStateProcessor.snrToCQI(urllcSINRdB);
                    urllcRate = Config.W * ChannelStateProcessor.cqiToEfficiency(urllcCQI);
                    [servedURLLCBits, servedDelay] = this.serveURLLCBits( ...
                        urllcRate * miniSlotDuration, ...
                        currentServiceStep, ...
                        miniSlotDuration);
                    remainingURLLCBits = remainingURLLCBits - servedURLLCBits;
                    maxURLLCDelay = max(maxURLLCDelay, servedDelay);
                end

                if thetaB > 0
                    embbSINR = ((1 - thetaB) * heb) / (noisePowerLinear + thetaB * heb + eps);
                else
                    embbSINR = heb / (noisePowerLinear + eps);
                end
                embbSINRdB = 10 * log10(max(eps, embbSINR));
                embbCQI = ChannelStateProcessor.snrToCQI(embbSINRdB);
                embbRate = Config.W * ChannelStateProcessor.cqiToEfficiency(embbCQI);

                embbBits = embbRate * miniSlotDuration;
                drainedEMBB = min(this.eMBBGroupQueues(groupId), embbBits);
                this.eMBBGroupQueues(groupId) = this.eMBBGroupQueues(groupId) - drainedEMBB;
                embbActualRates(rb) = drainedEMBB / miniSlotDuration;
            end

            this.URLLCGroupQueues = max(0.0, this.URLLCGroupQueues);
            this.eMBBGroupQueues = max(0.0, this.eMBBGroupQueues);

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
                embbActualRates, ...
                this.eMBBGroupQueues);

            if isnan(reward) || isinf(reward)
                fprintf('CRITICAL ERROR: Reward is %f at Global Step %d!\n', ...
                    reward, this.GlobalStepIndex);
            end
            if mod(this.CurrentStep, 200) == 0
                fprintf(['[Env Heartbeat] Ep Step: %d | URLLC Q: %.1f | ' ...
                    'eMBB Q: %.1f | Step Reward: %.4f\n'], ...
                    this.CurrentStep, ...
                    sum(this.URLLCGroupQueues), ...
                    sum(this.eMBBGroupQueues), ...
                    reward);
            end

            servedEmbbBits = sum(embbActualRates) * miniSlotDuration;
            totalEmbbQueue = sum(this.eMBBGroupQueues) + servedEmbbBits;
            if totalEmbbQueue > 0
                embbSatisfaction = servedEmbbBits / totalEmbbQueue;
            else
                embbSatisfaction = 1.0;
            end

            if all(embbActualRates == 0)
                embbFairness = 0.0;
            else
                fairnessDenominator = numel(embbActualRates) * sum(embbActualRates .^ 2);
                embbFairness = (sum(embbActualRates) ^ 2) / max(fairnessDenominator, eps);
            end

            isURLLCFailed = actualDelay > Config.tau_req;
            loggedSignals = struct( ...
                "urllc_actual_delay", actualDelay, ...
                "embb_satisfaction", embbSatisfaction, ...
                "embb_fairness", embbFairness, ...
                "is_urllc_failed", isURLLCFailed);

            newURLLCPackets = this.samplePoisson(Config.lambda_urllc, Config.N_g, 1);
            this.addURLLCPackets(newURLLCPackets, currentServiceStep + 1);

            this.GlobalStepIndex = this.GlobalStepIndex + 1;
            isDone = this.CurrentStep >= Config.Max_Episode_Steps;
            this.MiniSlotIndex = mod(this.CurrentStep, Config.M_mini_slots) + 1;

            if ~isDone
                safeIndex = mod(this.GlobalStepIndex - 1, this.MaxTraceSteps) + 1;
                this.URLLCChannelGain = this.TraceURLLC(:, safeIndex);
                this.eMBBChannelGain = this.TraceeMBB(:, safeIndex);
            end

            this.eMBBRateHistory = [this.eMBBRateHistory; mean(embbActualRates)];

            nextObservation = this.buildObservation();
            this.Observation = nextObservation;

            notifyEnvUpdated(this);
        end
    end

    methods (Access = private)
        function observation = buildObservation(this)
            normalizedURLLCQueues = min(1.0, max(0.0, this.URLLCGroupQueues / 10000.0));
            normalizedEMBBQueues = min(1.0, max(0.0, this.eMBBGroupQueues / 1e7));
            normalizedMiniSlotIndex = min(1.0, max(0.0, ...
                this.MiniSlotIndex / Config.M_mini_slots));

            observation = [
                normalizedURLLCQueues
                normalizedEMBBQueues
                normalizedMiniSlotIndex
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
            %SAMPLEPARETOBITS Draw a Pareto-distributed file size in bits.
            %   Uses inverse transform sampling to avoid toolbox dependence.

            if xm <= 0 || alpha <= 0
                error("RANSlicingEnv:InvalidParetoParameters", ...
                    "Pareto parameters xm and alpha must be positive.");
            end

            uniformSample = max(rand(), eps);
            sample = xm / (uniformSample ^ (1 / alpha));
            sample = min(sample, 200e6);
        end

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
