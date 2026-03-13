classdef Tests < matlab.unittest.TestCase
    %TESTS Unit tests for queue abstractions, reward logic, and Phase-1 PHY integration.
    %   Run all tests with:
    %       results = runtests('Tests.m');

    methods (Test)
        function testGroupUsersFixedBins(testCase)
            cqiArray = [15, 10, 7, 4, 1, 12, 9, 6, 3, 0, 14, 11, 8, 5, 2];
            groups = ChannelStateProcessor.groupUsers(cqiArray);

            testCase.verifyEqual(numel(groups), Config.N_g);
            testCase.verifyEqual(groups{1}, [1; 6; 11]);
            testCase.verifyEqual(groups{2}, [2; 7; 12]);
            testCase.verifyEqual(groups{3}, [3; 8; 13]);
            testCase.verifyEqual(groups{4}, [4; 9; 14]);
            testCase.verifyEqual(groups{5}, [5; 10; 15]);
        end

        function testQuantizeQueueEdgeCases(testCase)
            q0 = ChannelStateProcessor.quantizeQueue(0);
            qMid = ChannelStateProcessor.quantizeQueue(Config.phi_th / 2);
            qHigh = ChannelStateProcessor.quantizeQueue(Config.phi_th * 10);

            testCase.verifyEqual(q0, 0);
            testCase.verifyEqual(qMid, 6);
            testCase.verifyEqual(qHigh, Config.Q_levels - 1);
            testCase.verifyGreaterThanOrEqual(qMid, 0);
            testCase.verifyLessThanOrEqual(qMid, Config.Q_levels - 1);
        end

        function testNRPhyChannelStepProducesFiniteOutputs(testCase)
            phy = NRPhyChannel();
            phy.resetEpisode(101);

            metrics = phy.stepChannel(1, 1, 0.5 * ones(Config.N_g, 1), true);

            testCase.verifySize(metrics.urllcSignalPowerPerRB, [Config.B_RBs 1]);
            testCase.verifySize(metrics.embbSignalPowerPerRB, [Config.B_RBs 1]);
            testCase.verifyTrue(all(isfinite(metrics.urllcSignalPowerPerRB)));
            testCase.verifyTrue(all(isfinite(metrics.embbSignalPowerPerRB)));
            testCase.verifyTrue(all(isfinite(metrics.urllcCQIPerRB)));
            testCase.verifyTrue(all(isfinite(metrics.embbCQIPerRB)));
            testCase.verifyGreaterThanOrEqual(min(metrics.embbCQIPerRB), 0);
            testCase.verifyLessThanOrEqual(max(metrics.embbCQIPerRB), 15);
            testCase.verifyGreaterThanOrEqual(sum(metrics.groupURLLCPRBCount), 0);
            testCase.verifyLessThanOrEqual(sum(metrics.groupURLLCPRBCount), Config.B_RBs);
        end

        function testNRPhyChannelCalculateTBSMonotonicWithPRBs(testCase)
            phy = NRPhyChannel();
            smallTBS = phy.calculateTBS("64QAM", 666 / 1024.0, 1:2, [0, 14]);
            largeTBS = phy.calculateTBS("64QAM", 666 / 1024.0, 1:8, [0, 14]);

            testCase.verifyGreaterThan(smallTBS, 0);
            testCase.verifyGreaterThan(largeTBS, smallTBS);
        end

        function testNRPhyChannelQuotaExtremes(testCase)
            phy = NRPhyChannel();

            phy.resetEpisode(202);
            zeroQuota = phy.stepChannel(1, 1, zeros(Config.N_g, 1), true);

            phy.resetEpisode(202);
            fullQuota = phy.stepChannel(1, 1, ones(Config.N_g, 1), true);

            testCase.verifyEqual(sum(zeroQuota.groupURLLCPRBCount), 0);
            testCase.verifyEqual(sum(zeroQuota.groupEMBBPRBCount), Config.B_RBs);
            testCase.verifyEqual(sum(fullQuota.groupURLLCPRBCount), Config.B_RBs);
            testCase.verifyEqual(sum(fullQuota.groupEMBBPRBCount), 0);
        end

        function testCalculateRewardDelayViolationReturnsSoftPenalty(testCase)
            actualDelay = Config.tau_req * 2;
            reqDelay = Config.tau_req;
            embbRates = [10e6; 15e6; 20e6];
            embbQueueBits = [2e6; 1e6; 0.5e6];

            reward = calculateReward(actualDelay, reqDelay, embbRates, embbQueueBits);
            expectedPenalty = -1.0 * ((actualDelay - reqDelay) / reqDelay);
            expectedQueuePenalty = max( ...
                -50.0, ...
                -Config.embb_penalty_scale * sum(embbQueueBits));
            expectedReward = (expectedPenalty + expectedQueuePenalty) / Config.reward_scale;
            testCase.verifyEqual(reward, expectedReward, "AbsTol", 1e-12);
        end

        function testCalculateRewardZeroRatesFairnessEdgeCase(testCase)
            actualDelay = Config.tau_req * 0.5;
            reqDelay = Config.tau_req;
            embbRates = zeros(4, 1);
            embbQueueBits = zeros(4, 1);

            reward = calculateReward(actualDelay, reqDelay, embbRates, embbQueueBits);

            testCase.verifyTrue(isfinite(reward));
            % Build expected reward using Config constants to avoid brittle
            % numeric literals tied to internal scaling.
            expectedQueuePenalty = max(-50.0, -Config.embb_penalty_scale * sum(embbQueueBits));
            expectedReward = (expectedQueuePenalty + Config.reward_fairness_weight * 1.0) / Config.reward_scale;
            testCase.verifyEqual(reward, expectedReward, "AbsTol", 1e-12);
        end

        function testCalculateRewardNominalCase(testCase)
            actualDelay = Config.tau_req * 0.8;
            reqDelay = Config.tau_req;
            embbRates = [10e6; 20e6];
            embbQueueBits = [2e6; 1e6];

            reward = calculateReward(actualDelay, reqDelay, embbRates, embbQueueBits);

            fFairExpected = (sum(embbRates) ^ 2) / ...
                (numel(embbRates) * sum(embbRates .^ 2));
            expectedQueuePenalty = max( ...
                -50.0, ...
                -Config.embb_penalty_scale * sum(embbQueueBits));
            rewardExpected = (expectedQueuePenalty + Config.reward_fairness_weight * fFairExpected) / Config.reward_scale;

            testCase.verifyEqual(reward, rewardExpected, "AbsTol", 1e-12);
        end

        function testCalculateRewardDelayViolationCannotBecomePositive(testCase)
            actualDelay = 1.01 * Config.tau_req;
            embbRates = [1; 1];
            embbQueueBits = 0;
            reward = calculateReward(actualDelay, Config.tau_req, embbRates, embbQueueBits);

            expectedPenalty = -1.0 * ((actualDelay - Config.tau_req) / Config.tau_req);
            expectedQueuePenalty = max(-50.0, -Config.embb_penalty_scale * sum(embbQueueBits));
            expectedReward = (expectedPenalty + expectedQueuePenalty) / Config.reward_scale;

            testCase.verifyEqual(reward, expectedReward, "AbsTol", 1e-12);
            testCase.verifyLessThan(reward, 0.0);
        end

        function testResetProducesDeterministicPhyStateForFixedSeed(testCase)
            env = RANSlicingEnv();

            reset(env, 11);
            firstURLLCGain = env.URLLCChannelGain;
            firstEMBBCQI = env.eMBBPRBCQI;
            reset(env, 11);
            secondURLLCGain = env.URLLCChannelGain;
            secondEMBBCQI = env.eMBBPRBCQI;
            reset(env, 23);
            thirdURLLCGain = env.URLLCChannelGain;

            testCase.verifyEqual(firstURLLCGain, secondURLLCGain, "AbsTol", 1e-12);
            testCase.verifyEqual(firstEMBBCQI, secondEMBBCQI, "AbsTol", 1e-12);
            testCase.verifyGreaterThan(norm(firstURLLCGain - thirdURLLCGain), 0.0);
        end

        function testStepUsesAllPRBsForEMBBWhenURLLCQueueIsEmpty(testCase)
            env = RANSlicingEnv();
            reset(env);
            env.lambda_embb = 0.0;

            env.URLLCGroupQueues = zeros(Config.N_g, 1);
            env.eMBBGroupQueues = 1e6 * ones(Config.N_g, 1);
            preQueue = env.eMBBGroupQueues;

            [~, ~, ~, logged] = step(env, ones(Config.N_g, 1));

            servedBits = sum(preQueue) - sum(env.eMBBGroupQueues);
            testCase.verifyGreaterThan(servedBits, 0.0);
            testCase.verifyGreaterThan(logged.embb_tbs_bits, 0.0);
            testCase.verifyEqual(logged.urllc_actual_delay, 0.0, "AbsTol", 1e-12);
        end

        function testStepTracksAgedURLLCDelay(testCase)
            env = RANSlicingEnv();
            env.lambda_embb = 0.0;
            zeroAction = zeros(Config.N_g, 1);
            seed = Tests.findSeedWithURLLCBacklog(env, zeroAction);

            testCase.assertNotEmpty(seed, "Failed to find a seed with delayed URLLC backlog.");

            reset(env, seed);
            for stepIdx = 1:5
                step(env, zeroAction); %#ok<NASGU>
            end

            testCase.verifyGreaterThan(sum(env.URLLCGroupQueues), 0.0);

            [~, ~, ~, logged] = step(env, ones(Config.N_g, 1));

            miniSlotDuration = Config.Slot_duration / Config.M_mini_slots;
            testCase.verifyGreaterThan(logged.urllc_actual_delay, miniSlotDuration);
        end

        function testStepObservationUsesNormalizedState(testCase)
            env = RANSlicingEnv();
            reset(env, 7);
            env.lambda_embb = 0.0;

            env.URLLCGroupQueues = [2500; 12500; 0; 500; 20000];
            env.eMBBGroupQueues = [1e5; 5e6; 2.5e7; 0; 8e6];
            env.MiniSlotIndex = 3;
            env.URLLCWidebandCQI = 6;
            env.eMBBWidebandCQI = 12;

            [nextObservation, ~, ~, ~] = step(env, zeros(Config.N_g, 1));

            expectedObservation = [
                min(1.0, max(0.0, log10(1.0 + env.URLLCGroupQueues) / Config.obs_urlllc_log_scale))
                min(1.0, max(0.0, log10(1.0 + env.eMBBGroupQueues) / Config.obs_embb_log_scale))
                min(1.0, max(0.0, env.MiniSlotIndex / Config.M_mini_slots))
                min(1.0, max(0.0, env.URLLCWidebandCQI / 15.0))
                min(1.0, max(0.0, env.eMBBWidebandCQI / 15.0))
            ];

            testCase.verifyEqual(nextObservation, expectedObservation, "AbsTol", 1e-12);
            testCase.verifyGreaterThanOrEqual(min(nextObservation), 0.0);
            testCase.verifyLessThanOrEqual(max(nextObservation), 1.0);
        end
    end

    methods (Static, Access = private)
        function seed = findSeedWithURLLCBacklog(env, zeroAction)
            seed = [];
            for candidate = 1:50
                reset(env, candidate);
                for stepIdx = 1:5
                    step(env, zeroAction); %#ok<NASGU>
                end
                if sum(env.URLLCGroupQueues) > 0
                    seed = candidate;
                    return;
                end
            end
        end
    end
end
