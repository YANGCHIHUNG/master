function reward = calculateReward(actualDelay, reqDelay, embbRates, embbQoS)
%CALCULATEREWARD Compute multi-objective reward for RAN slicing.
%   reward = calculateReward(actualDelay, reqDelay, embbRates, embbQoS)
%
%   Inputs:
%   - actualDelay: Scalar URLLC delay in seconds.
%   - reqDelay: Scalar URLLC delay requirement in seconds.
%   - embbRates: Vector [N x 1] or [1 x N] of achieved eMBB rates (bps).
%   - embbQoS: Vector [N x 1] or [1 x N] of target eMBB QoS rates (bps).
%
%   Output:
%   - reward: Scalar reward. When delay is violated, reward is a negative
%     soft penalty proportional to violation magnitude.
%
%   Reward definition:
%   1) URLLC soft delay penalty:
%      if actualDelay > reqDelay, reward = -((actualDelay-reqDelay)/reqDelay)
%      and positive eMBB terms are skipped.
%   2) eMBB satisfaction F_sat':
%      x_i = min(1, r_i / QoS_i), F_sat' = mean(x_i).
%   3) Jain fairness F_fair':
%      F_fair' = (sum(x_i)^2) / (N * sum(x_i^2)); if all rates are zero,
%      F_fair' = 0.
%   4) Stability F_sta':
%      F_sta' = max(0, 1 - std(r)/mean(r)).
%   5) Composite:
%      reward = Config.omega_1*F_sat' + Config.omega_2*F_fair' + ...
%               Config.omega_3*F_sta'.

arguments
    actualDelay (1,1) double
    reqDelay (1,1) double
    embbRates {mustBeNumeric, mustBeVector}
    embbQoS {mustBeNumeric, mustBeVector}
end

if actualDelay > reqDelay
    penalty = -1.0 * ((actualDelay - reqDelay) / reqDelay);
    reward = penalty;
    return;
end

rates = max(0.0, double(embbRates(:)));
qos = max(eps, double(embbQoS(:)));
if numel(rates) ~= numel(qos)
    error("calculateReward:SizeMismatch", ...
        "embbRates and embbQoS must have identical lengths.");
end

if isempty(rates)
    reward = 0.0;
    return;
end

satisfactionRatios = min(1.0, rates ./ qos);
fSat = mean(satisfactionRatios);

if all(rates == 0)
    fFair = 0.0;
else
    numerator = sum(satisfactionRatios) ^ 2;
    denominator = numel(satisfactionRatios) * sum(satisfactionRatios .^ 2);
    if denominator <= eps
        fFair = 0.0;
    else
        fFair = numerator / denominator;
    end
end

rateMean = mean(rates);
if rateMean <= eps
    fSta = 0.0;
else
    rateStd = std(rates, 0);
    fSta = max(0.0, 1.0 - rateStd / rateMean);
end

reward = Config.omega_1 * fSat ...
    + Config.omega_2 * fFair ...
    + Config.omega_3 * fSta;

end
