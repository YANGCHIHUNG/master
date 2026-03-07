classdef Tests < matlab.unittest.TestCase
    %TESTS Unit tests for channel-state and reward utility modules.
    %   Run all tests with:
    %       results = runtests('Tests.m');

    methods (Test)
        function testGroupUsersFixedBins(testCase)
            %TESTGROUPUSERSFIXEDBINS Validate fixed MCS-to-group routing.
            %   Uses a synthetic MCS vector and verifies each user index
            %   is routed to the expected quality group.

            mcsArray = [25, 20, 13, 8, 2, 28, 18, 12, 6, 0, 24, 23, 17, 11, 5];
            groups = ChannelStateProcessor.groupUsers(mcsArray);

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

        function testSNRToMCSBoundsAndMonotonicity(testCase)
            %TESTSNRTOMCSBOUNDSANDMONOTONICITY Check mapping safety.
            %   Confirms output range [0, 28] and non-decreasing behavior
            %   for increasing SNR inputs.

            snr = [-20, -5, 0, 10, 25, 40];
            mcs = ChannelStateProcessor.snrToMCS(snr);

            testCase.verifyGreaterThanOrEqual(min(mcs), 0);
            testCase.verifyLessThanOrEqual(max(mcs), 28);
            testCase.verifyEqual(mcs(1), 0);
            testCase.verifyEqual(mcs(end), 28);
            testCase.verifyTrue(all(diff(mcs) >= 0));
        end

        function testCalculateRewardDelayViolationReturnsZero(testCase)
            %TESTCALCULATEREWARDDELAYVIOLATIONRETURNSZERO Enforce hard constraint.

            actualDelay = Config.tau_req * 2;
            reqDelay = Config.tau_req;
            embbRates = [10e6; 15e6; 20e6];
            embbQoS = [20e6; 20e6; 20e6];

            reward = calculateReward(actualDelay, reqDelay, embbRates, embbQoS);
            testCase.verifyEqual(reward, 0.0);
        end

        function testCalculateRewardZeroRatesFairnessEdgeCase(testCase)
            %TESTCALCULATEREWARDZERORATESFAIRNESSEDGECASE Check zero-rate safety.
            %   Ensures Jain fairness computation does not divide by zero
            %   or produce NaN/Inf when all rates are zero.

            actualDelay = Config.tau_req * 0.5;
            reqDelay = Config.tau_req;
            embbRates = zeros(4, 1);
            embbQoS = [10e6; 20e6; 30e6; 40e6];

            reward = calculateReward(actualDelay, reqDelay, embbRates, embbQoS);

            testCase.verifyTrue(isfinite(reward));
            testCase.verifyEqual(reward, 0.0);
        end

        function testCalculateRewardNominalCase(testCase)
            %TESTCALCULATEREWARDNOMINALCASE Validate composite reward formula.

            actualDelay = Config.tau_req * 0.8;
            reqDelay = Config.tau_req;
            embbRates = [10e6; 20e6];
            embbQoS = [20e6; 20e6];

            reward = calculateReward(actualDelay, reqDelay, embbRates, embbQoS);

            x = [0.5; 1.0];
            fSatExpected = mean(x);
            fFairExpected = (sum(x) ^ 2) / (numel(x) * sum(x .^ 2));
            fStaExpected = max(0.0, 1.0 - std(embbRates, 0) / mean(embbRates));
            rewardExpected = Config.omega_1 * fSatExpected ...
                + Config.omega_2 * fFairExpected ...
                + Config.omega_3 * fStaExpected;

            testCase.verifyEqual(reward, rewardExpected, "AbsTol", 1e-12);
        end
    end
end
