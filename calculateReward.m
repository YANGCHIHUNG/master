function reward = calculateReward(actualDelay, reqDelay, embbRates, embbQueueBits)
%CALCULATEREWARD Compute queue-aware reward for RAN slicing.
%   reward = calculateReward(actualDelay, reqDelay, embbRates, embbQueueBits)
%
%   Inputs:
%   - actualDelay: Scalar URLLC delay in seconds.
%   - reqDelay: Scalar URLLC delay requirement in seconds.
%   - embbRates: Vector [N x 1] or [1 x N] of achieved eMBB rates (bps).
%     When embbQueueBits is a vector, embbRates must be aligned to the
%     same eMBB groups.
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
%   3) Jain fairness incentive on active eMBB groups only:
%      active groups are those with queueBits > 0. If no active groups
%      remain, fairness is set to 1.0 to reward perfect queue clearance.
%   4) Composite:
%      reward = urllcReward + embbQueuePenalty when delay is violated,
%      otherwise reward = embbQueuePenalty + Config.reward_fairness_weight * embbFairness.
%   5) Final scaling: the composite reward is divided by Config.reward_scale
%      so that the final reward magnitudes are stable for SAC training.

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

if ~isscalar(queueBits) && numel(queueBits) ~= numel(rates)
    error("calculateReward:SizeMismatch", ...
        "embbRates and embbQueueBits must have the same number of elements.");
end

if isscalar(queueBits)
    if queueBits > 0
        activeRates = rates;
    else
        activeRates = zeros(0, 1);
    end
else
    % Only calculate fairness for groups that actually have data to transmit.
    activeIdx = queueBits > 0;
    activeRates = rates(activeIdx);
end

if isempty(activeRates) || sum(activeRates) == 0
    % If no queues are active, the system has been fully cleared.
    fFair = 1.0;
else
    numerator = sum(activeRates) ^ 2;
    denominator = numel(activeRates) * sum(activeRates .^ 2);
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
    % Use configurable fairness weight from Config instead of magic 0.1
    reward = embbQueuePenalty + Config.reward_fairness_weight * fFair;
end

% Scale (or clip) reward using configurable divisor to keep magnitudes
% suitable for SAC training. This replaces the previous hard-coded /100.
reward = reward / Config.reward_scale;

end
