classdef ChannelStateProcessor
    %CHANNELSTATEPROCESSOR Channel and queue abstraction utilities.
    %   This utility class provides:
    %   1) SINR-to-CQI mapping for 3GPP style link abstraction.
    %   2) CQI-to-spectral-efficiency mapping based on TS 38.214.
    %   3) User grouping by CQI quality into Config.N_g groups.
    %   3) Queue-length quantization for RL state compression.
    %
    %   All methods are static and can be called without object creation.

    methods (Static)
        function cqi = snrToCQI(snrDB)
            %SNRTOCQI Map SNR in dB to CQI index in [0, 15].
            %   cqi = ChannelStateProcessor.snrToCQI(snrDB)
            %
            %   Input:
            %   - snrDB: Numeric scalar/vector/matrix of SNR values (dB).
            %
            %   Output:
            %   - cqi: Same size as snrDB, integer-like double in
            %     [0, 15], where 0 means outage / no supported CQI.
            %
            %   Notes:
            %   - Thresholds approximate 10%% BLER switching points for
            %     CQI 1..15 in 5G NR style link abstraction.

            arguments
                snrDB {mustBeNumeric}
            end

            thresholds = [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, ...
                10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7];

            snr = double(snrDB);
            cqi = zeros(size(snr), "double");
            for level = 1:numel(thresholds)
                cqi(snr >= thresholds(level)) = level;
            end
        end

        function efficiency = cqiToEfficiency(cqi)
            %CQITOEFFICIENCY Map CQI index to 3GPP spectral efficiency.
            %   efficiency = ChannelStateProcessor.cqiToEfficiency(cqi)
            %
            %   Input:
            %   - cqi: Numeric scalar/vector/matrix of CQI values.
            %
            %   Output:
            %   - efficiency: Same size as cqi in bps/Hz.

            arguments
                cqi {mustBeNumeric}
            end

            seTable = [0.0, 0.1523, 0.2344, 0.3770, 0.6016, 0.8770, ...
                1.1758, 1.4766, 1.9141, 2.4063, 2.7305, 3.3223, ...
                3.9023, 4.5234, 5.1152, 5.5547];

            cqiIndex = round(double(cqi));
            cqiIndex = max(0, min(15, cqiIndex));
            efficiency = reshape(seTable(cqiIndex + 1), size(cqiIndex));
        end

        function groups = groupUsers(cqiArray)
            %GROUPUSERS Partition users into fixed CQI quality groups.
            %   groups = ChannelStateProcessor.groupUsers(cqiArray)
            %
            %   Input:
            %   - cqiArray: Numeric vector [N x 1] or [1 x N] containing CQI
            %     indices for N users.
            %
            %   Output:
            %   - groups: Cell array [Config.N_g x 1], where groups{k}
            %     contains 1-based user indices for group k:
            %       k=1: CQI 12-15 (Excellent)
            %       k=2: CQI  9-11 (Good)
            %       k=3: CQI  6-8  (Medium)
            %       k=4: CQI  3-5  (Poor)
            %       k=5: CQI  0-2  (Very Poor)

            arguments
                cqiArray {mustBeNumeric, mustBeVector}
            end

            % Enforce valid 3GPP CQI range.
            cqi = round(double(cqiArray(:)));
            cqi = max(0, min(15, cqi));

            ranges = [
                12, 15;
                9, 11;
                6, 8;
                3, 5;
                0, 2
            ];

            groups = cell(Config.N_g, 1);
            for groupId = 1:Config.N_g
                low = ranges(groupId, 1);
                high = ranges(groupId, 2);
                groups{groupId} = find(cqi >= low & cqi <= high);
            end
        end

        function q = quantizeQueue(queueLength)
            %QUANTIZEQUEUE Uniformly quantize queue length to state level.
            %   q = ChannelStateProcessor.quantizeQueue(queueLength)
            %
            %   Input:
            %   - queueLength: Numeric scalar/vector/matrix in bits.
            %
            %   Output:
            %   - q: Quantized queue level with same size as queueLength:
            %       0 for empty queue,
            %       1..(Config.Q_levels-2) for intermediate load,
            %       (Config.Q_levels-1) for saturation.
            %
            %   Formula:
            %   q = floor(queueLength / (phi_th / (Q_levels - 2))) + 1
            %   with explicit handling for queueLength == 0 and saturation.

            arguments
                queueLength {mustBeNumeric}
            end

            q = zeros(size(queueLength), "double");
            queueBits = max(0.0, double(queueLength));

            levelStep = Config.phi_th / (Config.Q_levels - 2);
            isZero = (queueBits == 0);
            isSaturated = (queueBits >= Config.phi_th);
            isMiddle = ~isZero & ~isSaturated;

            q(isMiddle) = floor(queueBits(isMiddle) ./ levelStep) + 1;
            q(isSaturated) = Config.Q_levels - 1;
            q = max(0, min(Config.Q_levels - 1, q));
        end
    end
end
