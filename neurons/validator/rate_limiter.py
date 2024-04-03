import time


class RateLimiter:
    def __init__(self, max_requests: int, period: float) -> None:
        """
        Initializes a new RateLimiter instance.

        Args:
            max_requests (int): The maximum number of requests that are allowed for a single key
                within the specified time period.

            period (int): The time period, in seconds, for which the request limit
                (specified by max_requests) applies.
        """
        self._max_requests = max_requests  # Max requests allowed
        self._period = period  # Time period in seconds
        self._requests: dict[str, tuple[int, float]] = {}

    def is_allowed(self, hotkey: str) -> bool:
        request_count, first_request_time = self._requests.get(hotkey, (0, 0.0))
        current_time = time.time()

        if current_time - first_request_time > self._period:
            self._requests[hotkey] = (1, current_time)
            return True

        if request_count >= self._max_requests:
            return False

        self._requests[hotkey] = (request_count + 1, first_request_time)
        return True
