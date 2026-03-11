function reward = calculateReward(actualDelay, reqDelay, embbRates, embbQueueBits)
%CALCULATEREWARD Compute queue-aware reward for RAN slicing.
%   reward = calculateReward(actualDelay, reqDelay, embbRates, embbQueueBits)
%
%   Inputs:
%   - actualDelay: Scalar URLLC delay in seconds.
%   - reqDelay: Scalar URLLC delay requirement in seconds.
%   - embbRates: Vector [N x 1] or [1 x N] of achieved eMBB rates (bps).
%   - embbQueueBits: Scalar or vector of remaining eMBB queue backlog (bits).
%
%   Output:
%   - reward: Scalar reward composed of URLLC delay penalty, eMBB queue
%     backlog penalty, and a small Jain fairness incentive.
%
%   Reward definition:
%   1) URLLC soft delay penalty:
%      urllcReward = -((actualDelay-reqDelay)/reqDelay) when violated,
%      otherwise 0.
%   2) eMBB queue backlog penalty:
%      embbQueuePenalty = -Config.embb_penalty_scale * sum(embbQueueBits).
%   3) Jain fairness incentive on achieved eMBB rates:
%      embbFairness = (sum(r)^2) / (N * sum(r.^2)); if all rates are zero,
%      embbFairness = 0.
%   4) Composite:
%      reward = urllcReward + embbQueuePenalty when delay is violated,
%      otherwise reward = embbQueuePenalty + 0.1 * embbFairness.

arguments
    actualDelay (1,1) double
    reqDelay (1,1) double
    embbRates {mustBeNumeric, mustBeVector}
    embbQueueBits {mustBeNumeric}
end

if actualDelay > reqDelay
    urllcReward = -1.0 * ((actualDelay - reqDelay) / reqDelay);
else
    urllcReward = 0.0;
end

rates = max(0.0, double(embbRates(:)));
queueBits = max(0.0, double(embbQueueBits(:)));

if isempty(rates)
    fFair = 0.0;
else
    numerator = sum(rates) ^ 2;
    denominator = numel(rates) * sum(rates .^ 2);
    if denominator <= eps
        fFair = 0.0;
    else
        fFair = numerator / denominator;
    end
end

embbQueuePenalty = -Config.embb_penalty_scale * sum(queueBits);
embbQueuePenalty = max(-50.0, embbQueuePenalty);

if actualDelay > reqDelay
    reward = urllcReward + embbQueuePenalty;
else
    reward = embbQueuePenalty + 0.1 * fFair;
end

end
