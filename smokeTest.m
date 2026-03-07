%SMOKETEST Environment API and runtime sanity validation.
%   This script validates the custom RL environment interface and executes
%   a random-action rollout to catch indexing, NaN, or divide-by-zero
%   errors before training.
%
%   Checks:
%   1) validateEnvironment(env) for RL Toolbox API compliance.
%   2) 140 random interaction steps with periodic reset on episode end.
%   3) Assertions ensuring finite observations, rewards, and logged fields.

clear;
clc;

env = RANSlicingEnv();
validateEnvironment(env);

obs = reset(env);
assert(all(isfinite(obs)), "smokeTest:InvalidObservation", ...
    "Initial observation contains NaN or Inf values.");

numSteps = Config.B_RBs * Config.M_mini_slots;
for stepIdx = 1:numSteps
    action = rand(Config.N_g, 1);
    [obs, reward, isDone, logged] = step(env, action);

    disp(["Step " + stepIdx, "Reward " + reward, ...
        "URLLC Failed " + string(logged.is_urllc_failed)]);

    assert(all(isfinite(obs)), "smokeTest:InvalidObservation", ...
        "Observation contains NaN or Inf values.");
    assert(isfinite(reward), "smokeTest:InvalidReward", ...
        "Reward is NaN or Inf.");
    assert(isfinite(logged.urllc_actual_delay), "smokeTest:InvalidDelay", ...
        "Logged URLLC delay is NaN or Inf.");
    assert(isfinite(logged.embb_satisfaction), "smokeTest:InvalidSatisfaction", ...
        "Logged eMBB satisfaction is NaN or Inf.");
    assert(isfinite(logged.embb_fairness), "smokeTest:InvalidFairness", ...
        "Logged eMBB fairness is NaN or Inf.");

    if isDone
        obs = reset(env);
        assert(all(isfinite(obs)), "smokeTest:InvalidResetObservation", ...
            "Reset observation contains NaN or Inf values.");
    end
end

disp("smokeTest completed without runtime errors.");
