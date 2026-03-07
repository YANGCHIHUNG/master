classdef ChannelStateProcessor
    %CHANNELSTATEPROCESSOR Channel and queue abstraction utilities.
    %   This utility class provides:
    %   1) SNR-to-MCS mapping for 5G NR style link abstraction.
    %   2) User grouping by MCS quality into Config.N_g groups.
    %   3) Queue-length quantization for RL state compression.
    %
    %   All methods are static and can be called without object creation.

    methods (Static)
        function mcsIndex = snrToMCS(snrDB)
            %SNRTOMCS Map SNR in dB to MCS index in [0, 28].
            %   mcsIndex = ChannelStateProcessor.snrToMCS(snrDB)
            %
            %   Input:
            %   - snrDB: Numeric scalar/vector/matrix of SNR values (dB).
            %
            %   Output:
            %   - mcsIndex: Same size as snrDB, integer-like double in
            %     [0, 28], where 0 means the lowest MCS.
            %
            %   Notes:
            %   - This mapping is a monotonic approximation suitable for
            %     simulation flow control without dependency on CQI tables.

            arguments
                snrDB {mustBeNumeric}
            end

            minSNRdB = -5.0;
            maxSNRdB = 25.0;
            maxMCS = 28;

            mcsIndex = floor((double(snrDB) - minSNRdB) ...
                ./ (maxSNRdB - minSNRdB) * (maxMCS + 1));
            mcsIndex = max(0, min(maxMCS, mcsIndex));
        end

        function groups = groupUsers(mcsArray)
            %GROUPUSERS Partition users into fixed MCS quality groups.
            %   groups = ChannelStateProcessor.groupUsers(mcsArray)
            %
            %   Input:
            %   - mcsArray: Numeric vector [N x 1] or [1 x N] containing MCS
            %     indices for N users.
            %
            %   Output:
            %   - groups: Cell array [Config.N_g x 1], where groups{k}
            %     contains 1-based user indices for group k:
            %       k=1: MCS 24-28 (Excellent)
            %       k=2: MCS 18-23 (Good)
            %       k=3: MCS 12-17 (Medium)
            %       k=4: MCS  6-11 (Poor)
            %       k=5: MCS  0-5  (Very Poor)

            arguments
                mcsArray {mustBeNumeric, mustBeVector}
            end

            % Enforce valid 5G NR style MCS range.
            mcs = round(double(mcsArray(:)));
            mcs = max(0, min(28, mcs));

            ranges = [
                24, 28;
                18, 23;
                12, 17;
                6, 11;
                0, 5
            ];

            groups = cell(Config.N_g, 1);
            for groupId = 1:Config.N_g
                low = ranges(groupId, 1);
                high = ranges(groupId, 2);
                groups{groupId} = find(mcs >= low & mcs <= high);
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
