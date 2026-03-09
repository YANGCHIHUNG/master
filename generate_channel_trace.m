%GENERATE_CHANNEL_TRACE Build offline channel traces for URLLC and eMBB.
%   This script generates a deterministic-size trace file:
%   - urllcChannelTrace: [Config.B_RBs x 100000]
%   - embbChannelTrace:  [Config.B_RBs x 100000]
%
%   Generation flow:
%   1) Use nrTDLChannel with TDL-A profile.
%   2) Per slot, estimate perfect channel on the full resource grid.
%   3) Average |h|^2 over 12 subcarriers (1 RB) and map symbols to mini-slots.
%   4) Create eMBB trace by circularly shifting URLLC trace.

rbCount = Config.B_RBs;
miniSlotsPerSlot = Config.M_mini_slots;
totalMiniSlots = 100000;

carrier = nrCarrierConfig;
carrier.NSizeGrid = rbCount;
carrier.SubcarrierSpacing = 15;

ofdmInfo = nrOFDMInfo(carrier);
symbolsPerSlot = ofdmInfo.SymbolsPerSlot;
samplesPerSlot = sum(ofdmInfo.SymbolLengths);

tdl = nrTDLChannel;
tdl.DelayProfile = "TDL-A";
tdl.DelaySpread = 30e-9;
tdl.MaximumDopplerShift = 10;
tdl.NumTransmitAntennas = 1;
tdl.NumReceiveAntennas = 1;
tdl.SampleRate = ofdmInfo.SampleRate;

pathFilters = getPathFilters(tdl);
numSlotsToGenerate = ceil(totalMiniSlots / miniSlotsPerSlot);

urllcChannelTrace = zeros(rbCount, totalMiniSlots, "double");
nextMiniSlot = 1;
dummyWaveform = zeros(samplesPerSlot, 1, "double");

for slotIdx = 1:numSlotsToGenerate
    [~, pathGains, sampleTimes] = tdl(dummyWaveform);
    channelGrid = nrPerfectChannelEstimate(carrier, pathGains, pathFilters, 0, sampleTimes);

    channelGrid = squeeze(channelGrid(:, :, 1, 1));
    powerGrid = abs(channelGrid).^2;

    for miniIdx = 1:miniSlotsPerSlot
        if nextMiniSlot > totalMiniSlots
            break;
        end

        firstSym = floor((miniIdx - 1) * symbolsPerSlot / miniSlotsPerSlot) + 1;
        lastSym = floor(miniIdx * symbolsPerSlot / miniSlotsPerSlot);
        lastSym = max(lastSym, firstSym);

        miniPowerPerSubcarrier = mean(powerGrid(:, firstSym:lastSym), 2);
        miniPowerPerRB = mean(reshape(miniPowerPerSubcarrier, 12, rbCount), 1).';
        urllcChannelTrace(:, nextMiniSlot) = miniPowerPerRB;

        nextMiniSlot = nextMiniSlot + 1;
    end
end

if nextMiniSlot <= totalMiniSlots
    error("generate_channel_trace:InsufficientSamples", ...
        "Trace generation stopped early. Expected %d mini-slots, got %d.", ...
        totalMiniSlots, nextMiniSlot - 1);
end

embbChannelTrace = circshift(urllcChannelTrace, 5, 2);
save("ChannelTraces.mat", "urllcChannelTrace", "embbChannelTrace");

disp("ChannelTraces.mat generated successfully.");
disp("urllcChannelTrace size:");
disp(size(urllcChannelTrace));
disp("embbChannelTrace size:");
disp(size(embbChannelTrace));
