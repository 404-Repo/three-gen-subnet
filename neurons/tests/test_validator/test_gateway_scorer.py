import time

import pytest
from validator.gateway.gateway import Gateway
from validator.gateway.gateway_api import GatewayApi
from validator.gateway.gateway_manager import GatewayManager
from validator.gateway.gateway_scorer import GatewayScorer


@pytest.fixture
def gateway_scorer() -> GatewayScorer:
    return GatewayScorer()


@pytest.fixture
def gateway_api() -> GatewayApi:
    return GatewayApi()


@pytest.fixture
def gateway_manager(gateway_scorer: GatewayScorer, gateway_api: GatewayApi) -> GatewayManager:
    return GatewayManager(
        gateway_scorer=gateway_scorer,
        gateway_api=gateway_api,
        gateway_info_server="https://test-gateway.404.xyz:4443",
    )


class TestGatewayScorer:

    def test_gateway_scorer_basic_scoring(self, gateway_scorer: GatewayScorer) -> None:
        """Test basic scoring functionality with two gateways."""
        # Create two gateways with different latencies and task counts
        gateway1 = Gateway(
            node_id=1,
            domain="gateway1.test",
            ip="1.1.1.1",
            name="Gateway 1",
            http_port=443,
            available_tasks=10,
            last_task_acquisition=int(time.time()),
            latency=100.0,
        )

        gateway2 = Gateway(
            node_id=2,
            domain="gateway2.test",
            ip="2.2.2.2",
            name="Gateway 2",
            http_port=443,
            available_tasks=5,
            last_task_acquisition=int(time.time()),
            latency=200.0,
        )

        # Score the gateways
        scored_gateways = gateway_scorer.score(gateways=[gateway1, gateway2])
        # Calculate expected scores
        # Gateway1: latency_score = 100 (max points as it has best latency)
        #          task_score = 50 (max points as it has most tasks)
        # Gateway2: latency_score = 10 (min points as latency is 2x worse)
        #          task_score = 25 (half points as it has half the tasks)
        assert scored_gateways[0].score == pytest.approx(150.0)  # 100 + 50
        assert scored_gateways[1].score == pytest.approx(35.0)  # 10 + 25
        # Gateway 1 should have a higher score due to lower latency and more tasks
        assert scored_gateways[0].score > scored_gateways[1].score
        assert scored_gateways[0].score > GatewayScorer.GATEWAY_MIN_SCORE
        assert scored_gateways[1].score > GatewayScorer.GATEWAY_MIN_SCORE

    def test_gateway_scorer_hungry_gateway(self, gateway_scorer: GatewayScorer) -> None:
        """Test scoring for a gateway that hasn't had tasks acquired for a while."""
        gateway = Gateway(
            node_id=1,
            domain="gateway.test",
            ip="1.1.1.1",
            name="Gateway",
            http_port=443,
            available_tasks=10,
            last_task_acquisition=0,  # This will make it considered "hungry"
            latency=100.0,
        )

        scored_gateways = gateway_scorer.score(gateways=[gateway])
        # Score should be HUNGRY_GATEWAY_MIN_SCORE (150) + latency_score (100) + task_score (50) = 300
        assert scored_gateways[0].score == pytest.approx(300.0)

    def test_gateway_scorer_no_tasks(self, gateway_scorer: GatewayScorer) -> None:
        """Test scoring for a gateway with no available tasks."""
        gateway = Gateway(
            node_id=1,
            domain="gateway.test",
            ip="1.1.1.1",
            name="Gateway",
            http_port=443,
            available_tasks=0,
            last_task_acquisition=0,
            latency=100.0,
        )

        scored_gateways = gateway_scorer.score(gateways=[gateway])
        assert scored_gateways[0].score == GatewayScorer.GATEWAY_MIN_SCORE

    def test_gateway_scorer_unavailable(self, gateway_scorer: GatewayScorer) -> None:
        """Test scoring for an unavailable gateway (latency is None)."""
        gateway = Gateway(
            node_id=1,
            domain="gateway.test",
            ip="1.1.1.1",
            name="Gateway",
            http_port=443,
            available_tasks=10,
            last_task_acquisition=0,
            latency=None,
        )

        scored_gateways = gateway_scorer.score(gateways=[gateway])
        assert scored_gateways[0].score == GatewayScorer.GATEWAY_MIN_SCORE

    def test_gateway_manager_update_gateways(self, gateway_manager: GatewayManager) -> None:
        """Test updating gateways with new information."""
        # Initial gateway
        initial_gateway = Gateway(
            node_id=1,
            domain="gateway.test",
            ip="1.1.1.1",
            name="Gateway",
            http_port=443,
            available_tasks=10,
            last_task_acquisition=0,
            latency=100.0,
        )

        # Update with new information
        updated_gateway = Gateway(
            node_id=1,
            domain="gateway.test",
            ip="1.1.1.1",
            name="Updated Gateway",
            http_port=443,
            available_tasks=20,
            last_task_acquisition=0,
        )

        # Add initial gateway to manager
        gateway_manager._gateways = [initial_gateway]

        # Update with new gateway information
        gateway_manager.update_gateways(gateways=[updated_gateway])

        # Check that the gateway was updated
        assert len(gateway_manager._gateways) == 1
        assert gateway_manager._gateways[0].name == "Updated Gateway"
        assert gateway_manager._gateways[0].available_tasks == 20
        assert gateway_manager._gateways[0].latency == 100.0  # Latency should be preserved
        # Test that score was calculated correctly
        assert (
            gateway_manager._gateways[0].score > GatewayScorer.GATEWAY_MIN_SCORE
        )  # Score should be recalculated and higher than min
        # Since this is the only gateway with tasks and latency, it should get max points
        expected_score = (
            GatewayScorer._HUNGRY_GATEWAY_MIN_SCORE
            + GatewayScorer._GATEWAY_TASK_COUNT_MAX_PTS
            + GatewayScorer._GATEWAY_LATENCY_MAX_PTS
        )
        assert abs(gateway_manager._gateways[0].score - expected_score) < 0.01  # Allow small floating point difference

    def test_gateway_manager_add_new_gateway(self, gateway_manager: GatewayManager) -> None:
        """Test adding a new gateway to the manager."""
        # Initial gateway
        initial_gateway = Gateway(
            node_id=1,
            domain="gateway1.test",
            ip="1.1.1.1",
            name="Gateway 1",
            http_port=443,
            available_tasks=10,
            last_task_acquisition=0,
            latency=100.0,
        )

        # New gateway to add
        new_gateway = Gateway(
            node_id=2,
            domain="gateway2.test",
            ip="2.2.2.2",
            name="Gateway 2",
            http_port=443,
            available_tasks=15,
            last_task_acquisition=0,
            latency=None,
        )

        # Add initial gateway to manager
        gateway_manager._gateways = [initial_gateway]

        # Update with new gateway
        gateway_manager.update_gateways(gateways=[initial_gateway, new_gateway])

        # Check that both gateways are present
        assert len(gateway_manager._gateways) == 2
        domains = {g.domain for g in gateway_manager._gateways}
        assert "gateway1.test" in domains
        assert "gateway2.test" in domains
        # Test that scores were calculated correctly
        # First gateway should get max points for latency and proportional points for tasks
        expected_score_1 = (
            GatewayScorer._HUNGRY_GATEWAY_MIN_SCORE
            + GatewayScorer._GATEWAY_LATENCY_MAX_PTS
            + GatewayScorer._GATEWAY_TASK_COUNT_MAX_PTS * (10 / 15)
        )
        # Second gateway has no latency so should get minimum score since latency is None
        expected_score_2 = GatewayScorer.GATEWAY_MIN_SCORE
        assert gateway_manager._gateways[0].score == pytest.approx(expected_score_1)
        assert gateway_manager._gateways[1].score == pytest.approx(expected_score_2)

    @pytest.mark.parametrize(
        "gateways,expected_best_idx,expected_scores",
        [
            # Example 1 from GatewayScorer docstring
            (
                [
                    Gateway(
                        node_id=1,
                        domain="g1",
                        ip="1.1.1.1",
                        name="g1",
                        http_port=1,
                        available_tasks=10,
                        last_task_acquisition=int(time.time()) - 1,
                        latency=100,
                    ),
                    Gateway(
                        node_id=2,
                        domain="g2",
                        ip="2.2.2.2",
                        name="g2",
                        http_port=2,
                        available_tasks=10,
                        last_task_acquisition=int(time.time()) - 10,
                        latency=200,
                    ),
                ],
                0,  # Expected best index (Gateway 1)
                [150, 60],  # Expected scores [G1, G2]
            ),
            # Example 2 from GatewayScorer docstring (score adjusted based on code logic)
            (
                [
                    Gateway(
                        node_id=1,
                        domain="g1",
                        ip="1.1.1.1",
                        name="g1",
                        http_port=1,
                        available_tasks=10,
                        last_task_acquisition=int(time.time()) - 1,
                        latency=100,
                    ),
                    Gateway(
                        node_id=2,
                        domain="g2",
                        ip="2.2.2.2",
                        name="g2",
                        http_port=2,
                        available_tasks=10,
                        last_task_acquisition=int(time.time()) - 40,
                        latency=150,
                    ),
                ],
                1,  # Expected best index (Gateway 2)
                # Expected scores [G1, G2]. G2 score is 150 (hungry) + 55 (latency) + 50 (tasks) = 255
                # Note: Docstring example calculation sums to 210, but description and code logic yield 255.
                [150, 255],
            ),
            # Example 3 from GatewayScorer docstring
            (
                [
                    Gateway(
                        node_id=1,
                        domain="g1",
                        ip="1.1.1.1",
                        name="g1",
                        http_port=1,
                        available_tasks=10,
                        last_task_acquisition=int(time.time()) - 1,
                        latency=100,
                    ),
                    Gateway(
                        node_id=2,
                        domain="g2",
                        ip="2.2.2.2",
                        name="g2",
                        http_port=2,
                        available_tasks=1000,
                        last_task_acquisition=int(time.time()) - 10,
                        latency=150,
                    ),
                ],
                1,  # Expected best index (Gateway 2)
                [100.5, 105],  # Expected scores [G1, G2]
            ),
            # Case: No tasks
            (
                [
                    Gateway(
                        node_id=1,
                        domain="g1",
                        ip="1.1.1.1",
                        name="g1",
                        http_port=1,
                        available_tasks=0,
                        last_task_acquisition=int(time.time()) - 1,
                        latency=100,
                    ),
                    Gateway(
                        node_id=2,
                        domain="g2",
                        ip="2.2.2.2",
                        name="g2",
                        http_port=2,
                        available_tasks=0,
                        last_task_acquisition=int(time.time()) - 10,
                        latency=200,
                    ),
                ],
                0,  # Index doesn't strictly matter as both are min score, max picks first
                [GatewayScorer.GATEWAY_MIN_SCORE, GatewayScorer.GATEWAY_MIN_SCORE],
            ),
            # Case: Latency None
            (
                [
                    Gateway(
                        node_id=1,
                        domain="g1",
                        ip="1.1.1.1",
                        name="g1",
                        http_port=1,
                        available_tasks=10,
                        last_task_acquisition=int(time.time()) - 1,
                        latency=None,
                    ),
                    Gateway(
                        node_id=2,
                        domain="g2",
                        ip="2.2.2.2",
                        name="g2",
                        http_port=2,
                        available_tasks=10,
                        last_task_acquisition=int(time.time()) - 10,
                        latency=200,
                    ),
                ],
                1,  # Gateway 2 should be selected
                [GatewayScorer.GATEWAY_MIN_SCORE, 150.0],  # G2: latency=100, tasks=50
            ),
        ],
    )
    def test_gateway_scorer_absolute_scores(
        self, gateways: list[Gateway], expected_best_idx: int, expected_scores: list[float]
    ) -> None:
        """Tests the absolute score calculation based on GatewayScorer docstring examples."""
        scorer = GatewayScorer()
        scored_gateways = scorer.score(gateways=gateways)

        # Check absolute scores
        assert len(scored_gateways) == len(expected_scores)
        assert scored_gateways[0].score == pytest.approx(expected_scores[0])
        assert scored_gateways[1].score == pytest.approx(expected_scores[1])

        # Verify the best gateway index calculation matches expectation
        # Handle cases where multiple gateways might have the same max score
        max_score = max(g.score for g in scored_gateways)
        indices_with_max_score = [i for i, g in enumerate(scored_gateways) if g.score == pytest.approx(max_score)]

        # If the expected best index corresponds to the max score, the test passes for best index logic.
        # Note: If multiple gateways share the max score, the `max()` function's choice isn't guaranteed,
        # but the score calculation itself is the primary focus here.
        assert expected_best_idx in indices_with_max_score
