from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class SRPTGlobalScheduler(BaseGlobalScheduler):
    """
    Shortest Remaining Processing Time (SRPT) global scheduler.
    Schedules requests based on their remaining token count (remaining work),
    with requests having the least remaining work getting highest priority.
    This is a preemptive version of SJF that considers work already done.
    Uses load balancing based on pending remaining tokens at each replica.
    """

    def schedule(self) -> List[Tuple[int, Request]]:
        # Sort requests by arrival time first (base behavior)
        self.sort_requests()

        # Sort by remaining tokens - least remaining work first
        self._request_queue.sort(
            key=lambda request: request.total_tokens - request.num_processed_tokens
        )

        request_mapping = []
        # Keep a map of replica_id -> pending remaining tokens
        # This tracks the current remaining work at each replica scheduler
        pending_remaining_tokens_map = {
            replica_scheduler.replica_id: replica_scheduler.num_pending_remaining_tokens
            for replica_scheduler in self._replica_schedulers.values()
        }

        while self._request_queue:
            request = self._request_queue.pop(0)
            remaining_tokens = request.total_tokens - request.num_processed_tokens
            
            # Find the replica with the least pending remaining tokens (least remaining work)
            replica_id = min(pending_remaining_tokens_map.items(), key=lambda x: x[1])[0]
            # Update the pending remaining tokens for this replica
            pending_remaining_tokens_map[replica_id] += remaining_tokens
            request_mapping.append((replica_id, request))

        return request_mapping
