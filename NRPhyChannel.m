classdef NRPhyChannel < handle
    %NRPHYCHANNEL Phase-1 5G Toolbox PHY abstraction for the RL environment.
    %   This helper class encapsulates:
    %   1) FR1 carrier configuration with 30 kHz SCS on a 20-PRB study grid.
    %   2) 3GPP TR 38.901 TDL-C fading plus scenario-based path loss.
    %   3) Wideband CQI estimation and CQI-to-MCS/TBS link adaptation.
    %   4) Group-wise PRB quota allocation for URLLC puncturing vs eMBB.

    properties
        Carrier
        OFDMInfo
        PathLossConfig
        URLLCChannel
        EMBBChannel
        CSIReportConfig
        PDSCHTemplate

        PathLossURLLCdB (1,1) double = Config.Path_Loss_dB
        PathLossEMBBdB (1,1) double = Config.Path_Loss_dB
        ShadowFadingURLLCdB (1,1) double = 0.0
        ShadowFadingEMBBdB (1,1) double = 0.0

        LastURLLCSignalPowerPerRB
        LastEMBBSignalPowerPerRB
        LastURLLCSINRdBPerRB
        LastEMBBSINRdBPerRB
        LastURLLCCQIPerRB
        LastEMBBCQIPerRB
        LastURLLCWidebandCQI (1,1) double = 0
        LastEMBBWidebandCQI (1,1) double = 0
        LastURLLCTBSBits (1,:) double = zeros(1, Config.N_g)
        LastEMBBTBSBits (1,:) double = zeros(1, Config.N_g)
        LastSymbolAllocation (1,2) double = [0, Config.Symbols_per_mini_slot]
        CurrentSlot (1,1) double = 0
    end

    properties (Access = private)
        LocalRandStream
    end

    methods
        function this = NRPhyChannel()
            %NRPHYCHANNEL Construct carrier, fading, and link-adaptation objects.

            this.Carrier = nrCarrierConfig;
            this.Carrier.NCellID = Config.nr_n_cell_id;
            this.Carrier.SubcarrierSpacing = Config.nr_subcarrier_spacing_khz;
            this.Carrier.CyclicPrefix = char(Config.nr_cyclic_prefix);
            this.Carrier.NSizeGrid = Config.nr_n_size_grid;
            this.Carrier.NStartGrid = Config.nr_n_start_grid;

            this.OFDMInfo = nrOFDMInfo(this.Carrier, ...
                "CarrierFrequency", Config.nr_center_frequency_hz);

            this.PathLossConfig = nrPathLossConfig( ...
                "Scenario", char(Config.channel_scenario));

            this.CSIReportConfig = nrCSIReportConfig;
            this.CSIReportConfig.NSizeBWP = Config.nr_n_size_grid;
            this.CSIReportConfig.NStartBWP = Config.nr_n_start_grid;
            this.CSIReportConfig.CQITable = char(Config.la_cqi_table);
            this.CSIReportConfig.CQIFormatIndicator = "wideband";

            this.PDSCHTemplate = nrPDSCHConfig;
            this.PDSCHTemplate.NSizeBWP = Config.nr_n_size_grid;
            this.PDSCHTemplate.NStartBWP = Config.nr_n_start_grid;
            this.PDSCHTemplate.NumLayers = Config.la_num_layers;
            this.PDSCHTemplate.MappingType = char(Config.la_pdsch_mapping_type);
            this.PDSCHTemplate.SymbolAllocation = Config.la_pdsch_symbol_allocation;

            this.LocalRandStream = RandStream("mt19937ar", ...
                "Seed", Config.channel_seed_base);

            this.URLLCChannel = this.createTDLChannel(Config.channel_seed_base + 1);
            this.EMBBChannel = this.createTDLChannel(Config.channel_seed_base + 2);

            this.resetEpisode(Config.channel_seed_base);
        end

        function resetEpisode(this, seedOrStream)
            %RESETEPISODE Reset fading/channel states for a new RL episode.

            % Accept either a numeric seed or a RandStream.
            if nargin < 2 || isempty(seedOrStream)
                seedValue = Config.channel_seed_base;
                this.LocalRandStream = RandStream("mt19937ar", "Seed", seedValue);
            elseif isa(seedOrStream, 'RandStream')
                this.LocalRandStream = seedOrStream;
                try
                    seedValue = this.LocalRandStream.Seed;
                catch
                    seedValue = Config.channel_seed_base;
                end
            else
                seedValue = max(0, round(double(seedOrStream)));
                this.LocalRandStream = RandStream("mt19937ar", "Seed", seedValue);
            end

            release(this.URLLCChannel);
            release(this.EMBBChannel);
            this.URLLCChannel.Seed = seedValue + 1;
            this.EMBBChannel.Seed = seedValue + 2;
            this.URLLCChannel.InitialTime = 0.0;
            this.EMBBChannel.InitialTime = 0.0;
            reset(this.URLLCChannel);
            reset(this.EMBBChannel);
            this.CurrentSlot = 0;

            [this.PathLossURLLCdB, this.ShadowFadingURLLCdB] = ...
                this.computePathLoss(Config.channel_ue_urlcc_position_m);
            [this.PathLossEMBBdB, this.ShadowFadingEMBBdB] = ...
                this.computePathLoss(Config.channel_ue_embb_position_m);

            zeroVector = zeros(Config.B_RBs, 1);
            this.LastURLLCSignalPowerPerRB = zeroVector;
            this.LastEMBBSignalPowerPerRB = zeroVector;
            this.LastURLLCSINRdBPerRB = zeroVector;
            this.LastEMBBSINRdBPerRB = zeroVector;
            this.LastURLLCCQIPerRB = zeroVector;
            this.LastEMBBCQIPerRB = zeroVector;
            this.LastURLLCWidebandCQI = 0;
            this.LastEMBBWidebandCQI = 0;
            this.LastURLLCTBSBits = zeros(1, Config.N_g);
            this.LastEMBBTBSBits = zeros(1, Config.N_g);
            this.LastSymbolAllocation = [0, Config.Symbols_per_mini_slot];
        end

        function metrics = stepChannel(this, stepIndex, miniSlotIndex, actionVec, hasURLLCBacklog)
            %STEPCHANNEL Advance fading and compute group-level PHY service.

            arguments
                this
                stepIndex (1,1) double
                miniSlotIndex (1,1) double
                actionVec {mustBeNumeric, mustBeVector}
                hasURLLCBacklog (1,1) logical
            end

            this.configureCarrierForStep(stepIndex);
            symbolAllocation = this.getMiniSlotSymbolAllocation(miniSlotIndex);

            [urllcResponse, ~] = this.URLLCChannel(this.Carrier);
            [embbResponse, ~] = this.EMBBChannel(this.Carrier);

            urllcSignalPowerPerRB = this.computeSignalPowerPerRB( ...
                urllcResponse, this.PathLossURLLCdB, symbolAllocation);
            embbSignalPowerPerRB = this.computeSignalPowerPerRB( ...
                embbResponse, this.PathLossEMBBdB, symbolAllocation);

            noisePowerLinear = 10 ^ (Config.Noise_Power / 10);
            urllcSINRLinear = urllcSignalPowerPerRB / max(noisePowerLinear, eps);
            embbSINRLinear = embbSignalPowerPerRB / max(noisePowerLinear, eps);
            urllcSINRdBPerRB = 10 * log10(max(urllcSINRLinear, eps));
            embbSINRdBPerRB = 10 * log10(max(embbSINRLinear, eps));

            urllcCQIPerRB = this.mapSINRToCQI(urllcSINRdBPerRB);
            embbCQIPerRB = this.mapSINRToCQI(embbSINRdBPerRB);
            groupMembers = ChannelStateProcessor.groupUsers(embbCQIPerRB);

            groupURLLCTBSBits = zeros(Config.N_g, 1);
            groupEMBBTBSBits = zeros(Config.N_g, 1);
            groupURLLCCQI = zeros(Config.N_g, 1);
            groupEMBBCQI = zeros(Config.N_g, 1);
            groupURLLCMCS = zeros(Config.N_g, 1);
            groupEMBBMCS = zeros(Config.N_g, 1);
            groupURLLCPRBCount = zeros(Config.N_g, 1);
            groupEMBBPRBCount = zeros(Config.N_g, 1);
            rbToGroup = ones(Config.B_RBs, 1) * Config.N_g;

            theta = min(1.0, max(0.0, double(actionVec(:))));
            for groupId = 1:Config.N_g
                groupRBs = groupMembers{groupId}(:).';
                rbToGroup(groupRBs) = groupId;
                if isempty(groupRBs)
                    continue;
                end

                if hasURLLCBacklog
                    requestedURLLCPRBs = round(theta(groupId) * numel(groupRBs));
                    requestedURLLCPRBs = min(numel(groupRBs), max(0, requestedURLLCPRBs));
                else
                    requestedURLLCPRBs = 0;
                end

                if requestedURLLCPRBs > 0
                    [~, priorityOrder] = sort(urllcSINRdBPerRB(groupRBs), "descend");
                    urllcRBs = groupRBs(priorityOrder(1:requestedURLLCPRBs));
                else
                    urllcRBs = zeros(1, 0);
                end
                embbRBs = setdiff(groupRBs, urllcRBs, "stable");

                groupURLLCPRBCount(groupId) = numel(urllcRBs);
                groupEMBBPRBCount(groupId) = numel(embbRBs);

                if ~isempty(urllcRBs)
                    groupURLLCCQI(groupId) = this.estimateCQI(urllcSINRdBPerRB, urllcRBs);
                    [groupURLLCMCS(groupId), modulation, targetCodeRate] = ...
                        this.mapCQIToMCS(groupURLLCCQI(groupId));
                    groupURLLCTBSBits(groupId) = this.calculateTBS( ...
                        modulation, targetCodeRate, urllcRBs, symbolAllocation);
                end

                if ~isempty(embbRBs)
                    groupEMBBCQI(groupId) = this.estimateCQI(embbSINRdBPerRB, embbRBs);
                    [groupEMBBMCS(groupId), modulation, targetCodeRate] = ...
                        this.mapCQIToMCS(groupEMBBCQI(groupId));
                    groupEMBBTBSBits(groupId) = this.calculateTBS( ...
                        modulation, targetCodeRate, embbRBs, symbolAllocation);
                end
            end

            urllcWidebandCQI = this.smoothCQI( ...
                this.LastURLLCWidebandCQI, ...
                this.estimateCQI(urllcSINRdBPerRB, 1:Config.B_RBs));
            embbWidebandCQI = this.smoothCQI( ...
                this.LastEMBBWidebandCQI, ...
                this.estimateCQI(embbSINRdBPerRB, 1:Config.B_RBs));

            miniSlotDuration = Config.Slot_duration / Config.M_mini_slots;
            metrics = struct( ...
                "symbolAllocation", symbolAllocation, ...
                "rbToGroup", rbToGroup, ...
                "groupMembers", {groupMembers}, ...
                "groupURLLCTBSBits", groupURLLCTBSBits, ...
                "groupEMBBTBSBits", groupEMBBTBSBits, ...
                "groupURLLCCQI", groupURLLCCQI, ...
                "groupEMBBCQI", groupEMBBCQI, ...
                "groupURLLCMCS", groupURLLCMCS, ...
                "groupEMBBMCS", groupEMBBMCS, ...
                "groupURLLCPRBCount", groupURLLCPRBCount, ...
                "groupEMBBPRBCount", groupEMBBPRBCount, ...
                "urllcSignalPowerPerRB", urllcSignalPowerPerRB, ...
                "embbSignalPowerPerRB", embbSignalPowerPerRB, ...
                "urllcSINRdBPerRB", urllcSINRdBPerRB, ...
                "embbSINRdBPerRB", embbSINRdBPerRB, ...
                "urllcCQIPerRB", urllcCQIPerRB, ...
                "embbCQIPerRB", embbCQIPerRB, ...
                "urllcWidebandCQI", urllcWidebandCQI, ...
                "embbWidebandCQI", embbWidebandCQI, ...
                "effectiveURLLCRatebps", groupURLLCTBSBits / miniSlotDuration, ...
                "effectiveEMBBRatebps", groupEMBBTBSBits / miniSlotDuration);

            this.LastSymbolAllocation = symbolAllocation;
            this.LastURLLCSignalPowerPerRB = urllcSignalPowerPerRB;
            this.LastEMBBSignalPowerPerRB = embbSignalPowerPerRB;
            this.LastURLLCSINRdBPerRB = urllcSINRdBPerRB;
            this.LastEMBBSINRdBPerRB = embbSINRdBPerRB;
            this.LastURLLCCQIPerRB = urllcCQIPerRB;
            this.LastEMBBCQIPerRB = embbCQIPerRB;
            this.LastURLLCWidebandCQI = urllcWidebandCQI;
            this.LastEMBBWidebandCQI = embbWidebandCQI;
            this.LastURLLCTBSBits = groupURLLCTBSBits.';
            this.LastEMBBTBSBits = groupEMBBTBSBits.';
        end

        function cqi = estimateCQI(this, sinrPerRBdB, prbIndices)
            %#ok<INUSD>
            if isempty(prbIndices)
                cqi = 0;
                return;
            end

            effectiveSINRdB = mean(double(sinrPerRBdB(prbIndices)));
            cqi = this.mapSINRToCQI(effectiveSINRdB);
        end

        function [mcsIndex, modulation, targetCodeRate] = mapCQIToMCS(this, cqi)
            %#ok<INUSD>
            cqiClamped = max(0, min(15, round(double(cqi))));
            modTable = [ ...
                "QPSK", "QPSK", "QPSK", "QPSK", "QPSK", "QPSK", ...
                "16QAM", "16QAM", "16QAM", ...
                "64QAM", "64QAM", "64QAM", "64QAM", "64QAM", "64QAM"];
            codeRateX1024 = [ ...
                78, 120, 193, 308, 449, 602, ...
                378, 490, 616, ...
                466, 567, 666, 772, 873, 948];

            if cqiClamped <= 0
                mcsIndex = 0;
                modulation = Config.la_modulation_default;
                targetCodeRate = 0.0;
                return;
            end

            mcsIndex = cqiClamped - 1;
            modulation = modTable(cqiClamped);
            targetCodeRate = codeRateX1024(cqiClamped) / 1024.0;
        end

        function tbsBits = calculateTBS(this, modulation, targetCodeRate, prbIndices, symbolAllocation)
            %CALCULATETBS Compute transport-block size for a PRB assignment.

            prbSet = unique(round(double(prbIndices(:).')));
            if isempty(prbSet) || targetCodeRate <= 0
                tbsBits = 0.0;
                return;
            end

            pdsch = this.PDSCHTemplate;
            pdsch.Modulation = char(modulation);
            pdsch.SymbolAllocation = double(symbolAllocation);
            pdsch.PRBSet = prbSet - 1;

            tbsBits = double(nrTBS( ...
                pdsch, ...
                targetCodeRate, ...
                Config.la_x_overhead, ...
                Config.la_tb_scaling));
        end

        function gains = getInstantaneousChannelGains(this, serviceType)
            service = upper(string(serviceType));
            switch service
                case "URLLC"
                    gains = this.LastURLLCSignalPowerPerRB(:);
                case "EMBB"
                    gains = this.LastEMBBSignalPowerPerRB(:);
                otherwise
                    error("NRPhyChannel:UnknownService", ...
                        "Unsupported service type %s.", serviceType);
            end
        end
    end

    methods (Access = private)
        function channel = createTDLChannel(this, seedValue)
            channel = nrTDLChannel;
            channel.DelayProfile = char(Config.channel_delay_profile);
            channel.DelaySpread = Config.channel_delay_spread_s;
            channel.MaximumDopplerShift = Config.channel_max_doppler_hz;
            channel.SampleRate = this.OFDMInfo.SampleRate;
            channel.NumTransmitAntennas = 1;
            channel.NumReceiveAntennas = 1;
            channel.NormalizePathGains = Config.channel_normalize_path_gains;
            channel.NormalizeChannelOutputs = Config.channel_normalize_outputs;
            channel.ChannelResponseOutput = "ofdm-response";
            channel.ChannelFiltering = false;
            channel.RandomStream = char(Config.channel_random_stream);
            channel.Seed = max(0, round(double(seedValue)));
        end

        function configureCarrierForStep(this, stepIndex)
            miniSlotStep = max(1, round(double(stepIndex)));
            slotIndex = floor((miniSlotStep - 1) / Config.M_mini_slots);
            slotsPerFrame = this.Carrier.SlotsPerFrame;
            this.Carrier.NFrame = floor(slotIndex / slotsPerFrame);
            this.Carrier.NSlot = mod(slotIndex, slotsPerFrame);
            this.CurrentSlot = slotIndex;
        end

        function symbolAllocation = getMiniSlotSymbolAllocation(this, miniSlotIndex)
            %#ok<INUSD>
            miniSlot = max(1, min(Config.M_mini_slots, round(double(miniSlotIndex))));
            startSymbol = (miniSlot - 1) * Config.Symbols_per_mini_slot;
            symbolAllocation = [startSymbol, Config.Symbols_per_mini_slot];
        end

        function signalPowerPerRB = computeSignalPowerPerRB(this, ofdmResponse, pathlossdB, symbolAllocation)
            startSymbol = symbolAllocation(1) + 1;
            endSymbol = startSymbol + symbolAllocation(2) - 1;
            responseSlice = squeeze(ofdmResponse(:, startSymbol:endSymbol, 1, 1));
            responseSlice = reshape(responseSlice, size(ofdmResponse, 1), []);

            perSubcarrierPower = mean(abs(responseSlice) .^ 2, 2);
            rbPower = reshape(perSubcarrierPower, Config.Subcarriers_per_RB, Config.B_RBs);
            meanRBPower = mean(rbPower, 1).';

            txPowerLinear = 10 ^ (Config.Tx_Power_dBm / 10);
            pathLossLinear = 10 ^ (pathlossdB / 10);
            signalPowerPerRB = max(eps, txPowerLinear * meanRBPower / max(pathLossLinear, eps));
        end

        function [pathlossdB, shadowDrawdB] = computePathLoss(this, uePosition)
            [basePathlossdB, shadowStdDevdB] = nrPathLoss( ...
                this.PathLossConfig, ...
                Config.nr_center_frequency_hz, ...
                Config.channel_is_los, ...
                Config.channel_bs_position_m, ...
                uePosition(:));

            if Config.channel_use_shadow_fading
                shadowDrawdB = shadowStdDevdB * randn(this.LocalRandStream, 1, 1);
            else
                shadowDrawdB = 0.0;
            end

            if Config.channel_use_pathloss
                pathlossdB = double(basePathlossdB) + double(shadowDrawdB);
            else
                pathlossdB = 0.0;
            end
        end

        function cqi = smoothCQI(this, previousCQI, rawCQI)
            if previousCQI <= 0
                cqi = rawCQI;
                return;
            end

            alpha = min(1.0, max(0.0, Config.la_cqi_smoothing_alpha));
            cqi = round(alpha * previousCQI + (1.0 - alpha) * rawCQI);
            cqi = max(0, min(15, cqi));
        end
    end

    methods (Static, Access = private)
        function cqi = mapSINRToCQI(sinrDB)
            thresholds = [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, ...
                10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7];

            sinr = double(sinrDB);
            cqi = zeros(size(sinr), "double");
            for thresholdIdx = 1:numel(thresholds)
                cqi(sinr >= thresholds(thresholdIdx)) = thresholdIdx;
            end
        end
    end
end
