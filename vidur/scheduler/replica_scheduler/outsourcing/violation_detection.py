"""
TTFT (Time-to-First-Token) violation detection for outsourcing decisions.
"""

from typing import Callable, List, Optional

from vidur import logger
from vidur.entities.batch import Request


class TTFTViolationDetector:
    """Detect imminent TTFT SLO violations."""

    def __init__(
        self,
        mode: str = "all",
        prefill_throughput: float = 1000.0,
        max_micro_batch_size: int = 8,
        ttft_tracker=None,  # Optional TTFTEstimateTracker instance
    ):
        """
        Initialize the violation detector.

        Args:
            mode: Detection mode ("all" or "head")
            prefill_throughput: Estimated prefill throughput (tokens/sec).
                               Can be updated later via set_prefill_throughput().
            max_micro_batch_size: Maximum micro-batch size
            ttft_tracker: Optional TTFTEstimateTracker instance for tracking estimates
        """
        self._mode = mode
        self._prefill_throughput = prefill_throughput
        self._max_micro_batch_size = max_micro_batch_size
        self._detector_func = self._get_detector_function(mode)
        self._ttft_tracker = ttft_tracker

    def set_prefill_throughput(self, throughput: float) -> None:
        """
        Update the prefill throughput estimate.
        
        Args:
            throughput: New prefill throughput estimate (tokens/sec)
        """
        self._prefill_throughput = throughput

    def _get_detector_function(self, mode: str) -> Callable:
        """Get the appropriate detector function based on mode."""
        detectors = {
            "all": self._check_all_violations,
            "head": self._check_head_violation,
        }
        if mode not in detectors:
            raise ValueError(
                f"Unknown TTFT violation mode: {mode}. "
                f"Choose from: {list(detectors.keys())}"
            )
        return detectors[mode]

    def check_violations(
        self,
        waiting_requests: List[Request],
        get_cached_length_func: Callable[[Request], int],
        current_time: float,
    ) -> bool:
        """
        Check if there are any TTFT violations.

        Args:
            waiting_requests: List of waiting requests
            get_cached_length_func: Function to get cached prefill length for a request
            current_time: Current simulation time

        Returns:
            True if violations detected, False otherwise
        """
        return self._detector_func(
            waiting_requests, get_cached_length_func, current_time
        )

    def _check_all_violations(
        self,
        waiting_requests: List[Request],
        get_cached_length_func: Callable[[Request], int],
        current_time: float,
    ) -> bool:
        """
        Check EVERY waiting request for imminent TTFT violation under FCFS.
        Returns True if any request is at risk of SLO violation.
        """
        if not waiting_requests:
            return False

        # Precompute remaining prefill for each request
        rem_prefill = []
        for r in waiting_requests:
            cached = get_cached_length_func(r)
            processed = r.num_processed_tokens
            prefill_done = max(processed, cached)
            rem = max(0, r.num_prefill_tokens - prefill_done)
            rem_prefill.append(rem)

        # Prefix sum: work ahead of each request in FCFS order
        ahead = [0] * len(waiting_requests)
        acc = 0
        for i in range(len(waiting_requests)):
            ahead[i] = acc
            acc += rem_prefill[i]

        # Evaluate every request with an SLO
        at_risk = set()
        saw_any_slo = False
        for i, r in enumerate(waiting_requests):
            if getattr(r, "prefill_slo_time", None) is None:
                continue
            saw_any_slo = True

            est_ttft = (ahead[i] + rem_prefill[i]) / self._prefill_throughput
            
            # Track the estimate if tracker is available
            if self._ttft_tracker is not None:
                deadline = getattr(r, "prefill_deadline_at", None)
                self._ttft_tracker.record_estimate(
                    request_id=r.id,
                    estimated_ttft=est_ttft,
                    current_time=current_time,
                    deadline=deadline,
                    remaining_prefill_tokens=rem_prefill[i],
                    queue_position=i,
                    ahead_prefill_tokens=ahead[i],
                )
            
            # print(f"Request {r.id}: est_ttft={est_ttft:.2f}s, "
            #             f"deadline_at={r.prefill_deadline_at:.2f}, "
            #             f"time_left={r.prefill_deadline_at - current_time:.2f}s")
            time_left = r.prefill_deadline_at - current_time
            if est_ttft > time_left:
                at_risk.add(r.id)

        # If none had an explicit SLO, fall back to a simple pressure heuristic
        if not saw_any_slo:
            if len(waiting_requests) > self._max_micro_batch_size:
                at_risk.update(r.id for r in waiting_requests[: self._max_micro_batch_size])

        return len(at_risk) > 0

    def _check_head_violation(
        self,
        waiting_requests: List[Request],
        get_cached_length_func: Callable[[Request], int],
        current_time: float,
    ) -> bool:
        """
        Check if the head request has imminent TTFT violation.
        Returns True if head request is at risk of SLO violation.
        """
        if not waiting_requests:
            return False

        head = waiting_requests[0]
        if head.prefill_slo_time is None:
            # Fallback to queue length heuristic
            return len(waiting_requests) > self._max_micro_batch_size

        est_ttft = self._estimate_fcfs_ttft(head, waiting_requests, get_cached_length_func)
        
        # Track the estimate if tracker is available
        if self._ttft_tracker is not None:
            deadline = getattr(head, "prefill_deadline_at", None)
            # Calculate remaining prefill for head
            cached = get_cached_length_func(head)
            processed = head.num_processed_tokens
            prefill_done = max(processed, cached)
            rem_prefill = max(0, head.num_prefill_tokens - prefill_done)
            
            self._ttft_tracker.record_estimate(
                request_id=head.id,
                estimated_ttft=est_ttft,
                current_time=current_time,
                deadline=deadline,
                remaining_prefill_tokens=rem_prefill,
                queue_position=0,
                ahead_prefill_tokens=0,  # Head has nothing ahead
            )
        
        return (head.prefill_deadline_at - current_time) < est_ttft

    def _estimate_fcfs_ttft(
        self,
        req: Request,
        waiting_requests: List[Request],
        get_cached_length_func: Callable[[Request], int],
    ) -> float:
        """
        Estimate Time-to-First-Token under FCFS assumption.
        Returns: queueing delay + own prefill time (in seconds).
        """
        if self._prefill_throughput <= 0:
            return float("inf")

        # Sum remaining prefill work of all waiting requests ahead of this one
        ahead_prefill = 0
        for r in waiting_requests:
            if r.id == req.id:
                break
            cached = get_cached_length_func(r)
            processed = r.num_processed_tokens
            prefill_done = max(processed, cached)
            rem = max(0, r.num_prefill_tokens - prefill_done)
            ahead_prefill += rem

        # Own prefill work
        cached = get_cached_length_func(req)
        processed = req.num_processed_tokens
        prefill_done = max(processed, cached)
        rem_self = max(0, req.num_prefill_tokens - prefill_done)

        # Convert to seconds
        est = (ahead_prefill + rem_self) / self._prefill_throughput
        return est
