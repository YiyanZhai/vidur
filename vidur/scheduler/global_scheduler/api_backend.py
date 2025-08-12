class ApiBackend:
    def __init__(self, fixed_overhead_ms: float = 400.0, per_token_ms: float = 0.0):
        self._fixed = fixed_overhead_ms
        self._per_token = per_token_ms

    def estimate_ttft_ms(self, request) -> float:
        # Simple parametric model—tune or replace later
        return self._fixed + self._per_token * max(1, request.num_decode_tokens)

    def schedule(self, now_s: float, request) -> float:
        # Return the absolute completion time (seconds) for an external API call
        ttft_ms = self.estimate_ttft_ms(request)
        # If you want only TTFT and then streaming, you can add per-token completion events;
        # minimal version: finish the whole request at once
        return now_s + ttft_ms / 1000.0
