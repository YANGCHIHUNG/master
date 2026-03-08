%% testAgent.m
% Diagnostic one-episode inference run for an existing rlSACAgent and environment.
% Assumes `agent`, `env`, and `Config` already exist in the current workspace.

% Basic sanity checks (no workspace clearing or reinitialization)
if ~exist('agent','var')
    error('Variable `agent` not found in workspace.');
end
if ~exist('env','var')
    error('Variable `env` not found in workspace.');
end

fprintf('\n===============================================================\n');
fprintf('Running ONE diagnostic episode for %d steps\n', Config.M_mini_slots);
fprintf('===============================================================\n\n');

obs = reset(env);
totalReward = 0;

for k = 1:Config.M_mini_slots
    actionCell = getAction(agent, obs);
    action = actionCell{1};

    [obs, reward, isDone, logged] = step(env, action);
    totalReward = totalReward + reward;

    theta = action(:).';
    urllcQ = env.URLLCGroupQueues(:).';

    fprintf('-------------------- STEP %4d --------------------\n', k);
    fprintf('theta (5):        [%.6f  %.6f  %.6f  %.6f  %.6f]\n', ...
        theta(1), theta(2), theta(3), theta(4), theta(5));
    fprintf('URLLC queues (5): [%.6f  %.6f  %.6f  %.6f  %.6f]\n', ...
        urllcQ(1), urllcQ(2), urllcQ(3), urllcQ(4), urllcQ(5));
    fprintf('reward: %.6f | logged.is_urllc_failed: %d | isDone: %d\n\n', ...
        reward, logged.is_urllc_failed, isDone);
end

fprintf('===============================================================\n');
fprintf('Total accumulated reward (one episode): %.6f\n', totalReward);
fprintf('===============================================================\n');
