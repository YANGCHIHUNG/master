classdef Tests < matlab.unittest.TestCase
    %TESTS Unit tests for channel-state and reward utility modules.
    %   Run all tests with:
    %       results = runtests('Tests.m');

    methods (Test)
        function testGroupUsersFixedBins(testCase)
            %TESTGROUPUSERSFIXEDBINS Validate fixed CQI-to-group routing.
            %   Uses a synthetic CQI vector and verifies each user index
            %   is routed to the expected quality group.

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
            %TESTQUANTIZEQUEUEEDGECASES Verify quantizer boundary behavior.
            %   Required checks:
            %   - queueLength = 0 returns 0.
            %   - queueLength = phi_th/2 remains in valid middle levels.
            %   - queueLength >> phi_th clamps to max level.

            q0 = ChannelStateProcessor.quantizeQueue(0);
            qMid = ChannelStateProcessor.quantizeQueue(Config.phi_th / 2);
            qHigh = ChannelStateProcessor.quantizeQueue(Config.phi_th * 10);

            testCase.verifyEqual(q0, 0);
            testCase.verifyEqual(qMid, 6);
            testCase.verifyEqual(qHigh, Config.Q_levels - 1);
            testCase.verifyGreaterThanOrEqual(qMid, 0);
            testCase.verifyLessThanOrEqual(qMid, Config.Q_levels - 1);
        end

        function testSNRToCQIBoundsAndMonotonicity(testCase)
            %TESTSNRTOCQIBOUNDSANDMONOTONICITY Check mapping safety.
            %   Confirms output range [0, 15] and non-decreasing behavior
            %   for increasing SNR inputs.

            snr = [-20, -6.7, 0, 10, 22.7, 40];
            cqi = ChannelStateProcessor.snrToCQI(snr);

            testCase.verifyGreaterThanOrEqual(min(cqi), 0);
            testCase.verifyLessThanOrEqual(max(cqi), 15);
            testCase.verifyEqual(cqi(1), 0);
            testCase.verifyEqual(cqi(2), 1);
            testCase.verifyEqual(cqi(end), 15);
            testCase.verifyTrue(all(diff(cqi) >= 0));
        end

        function testCQIToEfficiencyLookup(testCase)
            %TESTCQITOEFFICIENCYLOOKUP Validate 3GPP SE table endpoints.

            efficiency = ChannelStateProcessor.cqiToEfficiency([0, 15]);

            testCase.verifyEqual(efficiency(1), 0.0);
            testCase.verifyEqual(efficiency(2), 5.5547, "AbsTol", 1e-12);
        end

        function testCalculateRewardDelayViolationReturnsSoftPenalty(testCase)
            %TESTCALCULATEREWARDDELAYVIOLATIONRETURNSSOFTPENALTY Check new composite reward.

            actualDelay = Config.tau_req * 2;
            reqDelay = Config.tau_req;
            embbRates = [10e6; 15e6; 20e6];
            embbQueueBits = [2e6; 1e6; 0.5e6];

            reward = calculateReward(actualDelay, reqDelay, embbRates, embbQueueBits);
            expectedPenalty = -1.0 * ((actualDelay - reqDelay) / reqDelay);
            expectedQueuePenalty = max( ...
                -50.0, ...
                -Config.embb_penalty_scale * sum(embbQueueBits));
            expectedReward = (expectedPenalty + expectedQueuePenalty) / 100.0;
            testCase.verifyEqual(reward, expectedReward, "AbsTol", 1e-12);
        end

        function testCalculateRewardZeroRatesFairnessEdgeCase(testCase)
            %TESTCALCULATEREWARDZERORATESFAIRNESSEDGECASE Check zero-rate safety.
            %   Ensures Jain fairness computation does not divide by zero
            %   or produce NaN/Inf when all rates are zero.

            actualDelay = Config.tau_req * 0.5;
            reqDelay = Config.tau_req;
            embbRates = zeros(4, 1);
            embbQueueBits = zeros(4, 1);

            reward = calculateReward(actualDelay, reqDelay, embbRates, embbQueueBits);

            testCase.verifyTrue(isfinite(reward));
            testCase.verifyEqual(reward, 0.001, "AbsTol", 1e-12);
        end

        function testCalculateRewardNominalCase(testCase)
            %TESTCALCULATEREWARDNOMINALCASE Validate composite reward formula.

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
            rewardExpected = (expectedQueuePenalty + 0.1 * fFairExpected) / 100.0;

            testCase.verifyEqual(reward, rewardExpected, "AbsTol", 1e-12);
        end

        function testCalculateRewardDelayViolationCannotBecomePositive(testCase)
            %TESTCALCULATEREWARDDELAYVIOLATIONCANNOTBECOMEPOSITIVE Ensure fairness does not mask deadline misses.

            actualDelay = 1.01 * Config.tau_req;
            reward = calculateReward(actualDelay, Config.tau_req, [1; 1], 0);

            testCase.verifyEqual(reward, -0.0001, "AbsTol", 1e-12);
            testCase.verifyLessThan(reward, 0.0);
        end

        function testResetRandomizesTraceStartWithinSafeBounds(testCase)
            %TESTRESETRANDOMIZESTRACESTARTWITHINSAFEBOUNDS Verify reset picks a bounded random trace index.

            env = RANSlicingEnv();
            traceData = load("ChannelTraces.mat");
            maxStartIndex = size(traceData.urllcChannelTrace, 2) - ...
                Config.Max_Episode_Steps * Config.M_mini_slots - 1;

            if maxStartIndex <= 1
                rng(11);
                reset(env);
                testCase.verifyEqual(env.URLLCChannelGain, traceData.urllcChannelTrace(:, 1));
                testCase.verifyEqual(env.eMBBChannelGain, traceData.embbChannelTrace(:, 1));
                return;
            end

            firstSeed = 11;
            secondSeed = 23;

            rng(firstSeed);
            expectedFirstIndex = randi([1, maxStartIndex]);
            rng(firstSeed);
            reset(env);
            firstURLLCGain = env.URLLCChannelGain;
            firstEMBBGain = env.eMBBChannelGain;

            testCase.verifyEqual(firstURLLCGain, traceData.urllcChannelTrace(:, expectedFirstIndex));
            testCase.verifyEqual(firstEMBBGain, traceData.embbChannelTrace(:, expectedFirstIndex));

            rng(secondSeed);
            expectedSecondIndex = randi([1, maxStartIndex]);
            rng(secondSeed);
            reset(env);

            testCase.verifyEqual(env.URLLCChannelGain, traceData.urllcChannelTrace(:, expectedSecondIndex));
            testCase.verifyEqual(env.eMBBChannelGain, traceData.embbChannelTrace(:, expectedSecondIndex));
            testCase.verifyNotEqual(expectedFirstIndex, expectedSecondIndex);
        end

        function testStepUsesAllRBsWhenURLLCQueueIsEmpty(testCase)
            %TESTSTEPUSESALLRBSWHENURLLCQUEUEISEMPTY Ensure eMBB uses all RBs without phantom URLLC interference.

            env = RANSlicingEnv();
            reset(env);

            noisePowerLinear = 10 ^ (Config.Noise_Power / 10);
            rxPowerFactor = 10 ^ ((Config.Tx_Power_dBm - Config.Path_Loss_dB) / 10);
            snrDb = 10 * log10(rxPowerFactor / noisePowerLinear);
            cqi = ChannelStateProcessor.snrToCQI(snrDb);
            groupId = Tests.groupIdForCQI(cqi);
            perRbBits = Config.W * ChannelStateProcessor.cqiToEfficiency(cqi) * ...
                (Config.Slot_duration / Config.M_mini_slots);

            env.URLLCGroupQueues = zeros(Config.N_g, 1);
            env.eMBBGroupQueues = zeros(Config.N_g, 1);
            env.eMBBGroupQueues(groupId) = 1e6;
            env.URLLCChannelGain = ones(Config.B_RBs, 1);
            env.eMBBChannelGain = ones(Config.B_RBs, 1);

            seed = Tests.findSeedWithoutEmbbArrivals(200);
            testCase.assertNotEmpty(seed, "Failed to find a seed without eMBB arrivals.");
            rng(seed);

            [~, ~, ~, logged] = step(env, ones(Config.N_g, 1));

            expectedServedBits = Config.B_RBs * perRbBits;
            servedBits = 1e6 - env.eMBBGroupQueues(groupId);
            testCase.verifyEqual(servedBits, expectedServedBits, "AbsTol", 1e-9);
            testCase.verifyEqual(logged.urllc_actual_delay, 0.0, "AbsTol", 1e-12);
        end

        function testStepTracksAgedURLLCDelay(testCase)
            %TESTSTEPTRACKSAGEDURLLCDELAY Ensure old URLLC backlog reports aged delay when eventually served.

            env = RANSlicingEnv();
            seed = [];
            zeroAction = zeros(Config.N_g, 1);

            for candidate = 1:50
                reset(env);
                rng(candidate);
                for stepIdx = 1:5
                    step(env, zeroAction); %#ok<NASGU>
                end
                if sum(env.URLLCGroupQueues) > 0
                    seed = candidate;
                    break;
                end
            end

            testCase.assertNotEmpty(seed, "Failed to find a seed with delayed URLLC backlog.");

            reset(env);
            rng(seed);
            for stepIdx = 1:5
                step(env, zeroAction); %#ok<NASGU>
            end

            testCase.verifyGreaterThan(sum(env.URLLCGroupQueues), 0.0);

            env.URLLCChannelGain = ones(Config.B_RBs, 1);
            env.eMBBChannelGain = 0.1 * ones(Config.B_RBs, 1);
            [~, ~, ~, logged] = step(env, ones(Config.N_g, 1));

            miniSlotDuration = Config.Slot_duration / Config.M_mini_slots;
            testCase.verifyGreaterThan(logged.urllc_actual_delay, miniSlotDuration);
            testCase.verifyGreaterThan(logged.urllc_actual_delay, Config.tau_req);
        end

        function testStepObservationUsesNormalizedState(testCase)
            %TESTSTEPOBSERVATIONUSESNORMALIZEDSTATE Ensure returned observation matches normalized environment state.

            env = RANSlicingEnv();
            reset(env);
            rng(7);

            env.URLLCGroupQueues = [2500; 12500; 0; 500; 20000];
            env.eMBBGroupQueues = [1e5; 5e6; 2.5e7; 0; 8e6];
            env.MiniSlotIndex = 3;
            env.URLLCChannelGain = linspace(0.2, 1.2, Config.B_RBs).';
            env.eMBBChannelGain = linspace(1.2, 0.2, Config.B_RBs).';

            [nextObservation, ~, ~, ~] = step(env, zeros(Config.N_g, 1));

            expectedObservation = [
                min(1.0, max(0.0, log10(1.0 + env.URLLCGroupQueues) / 6.0))
                min(1.0, max(0.0, log10(1.0 + env.eMBBGroupQueues) / 9.0))
                min(1.0, max(0.0, env.MiniSlotIndex / Config.M_mini_slots))
            ];

            testCase.verifyEqual(nextObservation, expectedObservation, "AbsTol", 1e-12);
            testCase.verifyGreaterThanOrEqual(min(nextObservation), 0.0);
            testCase.verifyLessThanOrEqual(max(nextObservation), 1.0);
        end
    end

    methods (Static, Access = private)
        function seed = findSeedWithoutEmbbArrivals(maxSeeds)
            seed = [];
            for candidate = 1:maxSeeds
                rng(candidate);
                if all(rand(Config.N_g, 1) >= Config.lambda_embb)
                    seed = candidate;
                    return;
                end
            end
        end

        function groupId = groupIdForCQI(cqi)
            groups = ChannelStateProcessor.groupUsers(cqi);
            groupId = find(cellfun(@(idx) ~isempty(idx), groups), 1, "first");
        end
    end
end
