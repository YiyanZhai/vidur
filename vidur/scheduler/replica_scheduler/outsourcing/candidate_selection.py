"""
Candidate selection for outsourcing decisions.
"""

from typing import Callable, List, Set

from vidur.entities.batch import Request


class CandidateSelector:
    """Select candidate requests for outsourcing consideration."""

    def __init__(self):
        """Initialize the candidate selector."""
        pass

    def collect_candidates(
        self,
        waiting_requests: List[Request],
        running_requests: List[Request],
        outsourced_req_ids: Set[str],
        scheduled_req_ids: Set[str],
    ) -> List[Request]:
        """
        Collect requests eligible for outsourcing.

        Includes:
        - Waiting requests
        - Running prefill requests (avoid decode-phase ejection)

        Excludes:
        - Already outsourced requests
        - Currently scheduled requests
        - Requests in decode phase

        Args:
            waiting_requests: List of waiting requests
            running_requests: List of running requests
            outsourced_req_ids: Set of already outsourced request IDs
            scheduled_req_ids: Set of currently scheduled request IDs

        Returns:
            List of candidate requests for outsourcing
        """
        candidates = []

        # Get waiting requests
        for r in waiting_requests:
            if r.id not in outsourced_req_ids:
                candidates.append(r)

        # Add running prefill requests (avoid decode-phase ejection)
        for r in running_requests:
            if (
                not r.is_prefill_complete
                and r.id not in outsourced_req_ids
                and r.id not in scheduled_req_ids
            ):
                candidates.append(r)

        return candidates
