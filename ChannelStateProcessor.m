classdef ChannelStateProcessor
    %CHANNELSTATEPROCESSOR Channel and queue abstraction utilities.
    %   This utility class provides:
    %   1) User grouping by CQI quality into Config.N_g groups.
    %   2) Queue-length quantization for RL state compression.
    %
    %   All methods are static and can be called without object creation.

    methods (Static)
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
