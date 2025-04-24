import time

from validator.gateway.gateway import Gateway


class GatewayScorer:
    """Calculates score for gateways. Validator selects gateway with the highest score.

    EXAMPLE 1:
        Gateway 1:
            latency: 100 ms
            task_count: 10
            last_task_acquisition: 1 seconds ago
        Gateway 2:
            latency: 200 ms
            task_count: 10
            last_task_acquisition: 10 seconds ago
        Score:
            Latency score. Gateway_1 is 2 times better than gateway_2. Therefore gateway_1
            gets 100 points and gateway_2 gets 10 points.
            Task count score is the same for both gateways - 50 points.
            No one gateway is hungry (30 seconds threshold). Both get 0 points.
            Gateway 1 score: 100 + 50 + 0 = 150
            Gateway 2 score: 10 + 50 + 0 = 60
        Result:
            Select Gateway 1

    EXAMPLE 2:
        Gateway 1:
            latency: 100 ms
            task_count: 10
            last_task_acquisition: 1 seconds ago
        Gateway 2:
            latency: 150 ms
            task_count: 10
            last_task_acquisition: 40 seconds ago
        Score:
            Latency score. Gateway_1 is better than gateway_2. Therefore gateway_1
            gets 100 points and gateway_2 gets 10 + 50/100 * 90 = 55 points.
            Task count score is the same for both gateways - 50 points.
            Gateway 2 is hungry (exceeds 30 seconds threshold). It gets 150 points.
            Gateway 1 score: 100 + 50 + 0 = 150
            Gateway 2 score: 10 + 50 + 150 = 210
        Result:
            Select Gateway 2

    EXAMPLE 3:
        Gateway 1:
            latency: 100 ms
            task_count: 10
            last_task_acquisition: 1 seconds ago
        Gateway 2:
            latency: 150 ms
            task_count: 1000
            last_task_acquisition: 10 seconds ago
        Score:
            Latency score. Gateway_1 is better than gateway_2. Therefore gateway_1
            gets 100 points and gateway_2 gets 10 + 50/100 * 90 = 55 points.
            Task count score. Gateway_1 has 10 tasks and gateway_2 has 1000 tasks.
            Therefore gateway_2 gets 50 points and gateway_1 gets 0.5 points.
            No one gateway is hungry (30 seconds threshold). Both get 0 points.
            Gateway 1 score: 100 + 0.5 + 0 = 100.5
            Gateway 2 score: 55 + 50 + 0 = 105
        Result:
            Select Gateway 2
    """

    _GATEWAY_LATENCY_MAX_PTS: float = 100.0
    """Maximum points that gateway can get for latency."""
    _GATEWAY_TASK_COUNT_MAX_PTS: float = 50.0
    """Maximum points that gateway can get for task count."""
    _HUNGRY_GATEWAY_MIN_SCORE: float = 150.0
    """Minimum score for hungry gateway that didn't provided tasks for a while."""
    GATEWAY_MIN_SCORE: float = -1.0
    """Minimum score for gateway that is not available or has no tasks."""
    _GATEWAY_LAST_TASK_ACQUISITION_THRESHOLD_SECONDS: float = 30
    """Threshold for last task acquisition.
    If gateway has tasks but didn't acquire them for a while, it is considered hungry."""

    def score(self, *, gateways: list[Gateway]) -> list[Gateway]:
        latencies: list[float] = [gateway.latency for gateway in gateways if gateway.latency is not None]
        min_latency = min(latencies) if latencies else None
        task_counts: list[int] = [gateway.available_tasks for gateway in gateways]
        max_task_count: int = max(task_counts)

        for gateway in gateways:
            if gateway.disabled:
                # Gateway is on cooldown.
                gateway.score = self.GATEWAY_MIN_SCORE
                continue
            if gateway.available_tasks == 0:
                # No tasks available.
                gateway.score = self.GATEWAY_MIN_SCORE
                continue
            if gateway.latency is None:
                # Gateway is unavailable.
                gateway.score = self.GATEWAY_MIN_SCORE
                continue
            if time.time() - gateway.last_task_acquisition > self._GATEWAY_LAST_TASK_ACQUISITION_THRESHOLD_SECONDS:
                # Gateway has tasks but tasks are not acquired for a while.
                gateway.score = self._HUNGRY_GATEWAY_MIN_SCORE
            else:
                # Gateway has tasks and tasks are acquired recently.
                gateway.score = 0

            latency_score = self._calculate_latency_score(latency=gateway.latency, min_latency=min_latency)  # type: ignore
            task_count_score = self._calculate_task_count_score(
                task_count=gateway.available_tasks, max_task_count=max_task_count
            )
            gateway.score += latency_score + task_count_score
        return gateways

    def _calculate_latency_score(self, *, latency: float, min_latency: float) -> float:
        """
        Gateway with best latency gets max points.
        Gateway with two times worse latency and less gets (max_points / 10) points.
        Gateway with the latency in the middle gets proportional number of points.
        """
        bad_latency = min_latency * 2
        min_pts = self._GATEWAY_LATENCY_MAX_PTS * 0.1
        if latency > bad_latency:
            return min_pts
        return min_pts + self._GATEWAY_LATENCY_MAX_PTS * 0.9 * (bad_latency - latency) / (bad_latency - min_latency)

    def _calculate_task_count_score(self, *, task_count: int, max_task_count: int) -> float:
        """
        Points are proportional to the number of tasks.
        """
        if not max_task_count:
            return self.GATEWAY_MIN_SCORE
        return self._GATEWAY_TASK_COUNT_MAX_PTS * task_count / max_task_count


gateway_scorer = GatewayScorer()
