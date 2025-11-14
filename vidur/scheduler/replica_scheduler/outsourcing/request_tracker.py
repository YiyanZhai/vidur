"""
Request tracking and metrics collection for outsourcing.
"""

from typing import Callable, List

from vidur.entities.batch import Request


class RequestTracker:
    """Track outsourced requests and collect metrics."""

    def __init__(self, replica_id: int, cost_calculator: Callable):
        """
        Initialize the request tracker.

        Args:
            replica_id: ID of the replica
            cost_calculator: Function to calculate API cost (input_tokens, output_tokens) -> cost
        """
        self._replica_id = replica_id
        self._cost_calculator = cost_calculator
        self._outsourced_request_details: List[dict] = []

    def track_outsourced_request(
        self,
        request: Request,
        was_running: bool,
        current_time: float,
    ) -> None:
        """
        Track details of an outsourced request for later reporting.

        Args:
            request: The outsourced request
            was_running: Whether the request was running (vs waiting)
            current_time: Current simulation time
        """
        input_tokens = request.num_prefill_tokens
        output_tokens = request.num_decode_tokens
        api_cost = self._cost_calculator(input_tokens, output_tokens)

        self._outsourced_request_details.append(
            {
                "request_id": request.id,
                "outsourced_at": current_time,
                "arrived_at": request.arrived_at,
                "queued_at": request.queued_at,
                "was_running": was_running,
                "num_prefill_tokens": input_tokens,
                "num_decode_tokens": output_tokens,
                "num_processed_tokens": request.num_processed_tokens,
                "api_cost_usd": api_cost,
                "replica_id": str(self._replica_id),
            }
        )

    def get_outsourced_request_details(self) -> List[dict]:
        """Return the list of outsourced request details."""
        return self._outsourced_request_details

    def get_outsourcing_statistics(self) -> dict:
        """Calculate and return outsourcing statistics."""
        if not self._outsourced_request_details:
            return {
                "total_outsourced": 0,
                "outsourced_from_waiting": 0,
                "outsourced_from_running": 0,
                "total_api_cost_usd": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "replica_id": str(self._replica_id),
            }

        total = len(self._outsourced_request_details)
        from_running = sum(
            1 for d in self._outsourced_request_details if d["was_running"]
        )
        from_waiting = total - from_running
        total_cost = sum(
            d["api_cost_usd"] for d in self._outsourced_request_details
        )
        total_input = sum(
            d["num_prefill_tokens"] for d in self._outsourced_request_details
        )
        total_output = sum(
            d["num_decode_tokens"] for d in self._outsourced_request_details
        )

        return {
            "total_outsourced": total,
            "outsourced_from_waiting": from_waiting,
            "outsourced_from_running": from_running,
            "total_api_cost_usd": total_cost,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "replica_id": str(self._replica_id),
        }
