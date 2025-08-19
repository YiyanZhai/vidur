from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class SRPTGlobalScheduler(BaseGlobalScheduler):
    """
    Shortest Remaining Processing Time (SRPT) global scheduler.
    Schedules requests based on their remaining token count (remaining work),
    with requests having the least remaining work getting highest priority.
    This is a preemptive version of SJF that considers work already done.
    """

    def schedule(self) -> List[Tuple[int, Request]]:
        # Sort requests by arrival time first (base behavior)
        self.sort_requests()

        # Sort by remaining tokens - least remaining work first
        self._request_queue.sort(
            key=lambda request: request.total_tokens - request.num_processed_tokens
        )

        request_mapping = []
        replica_loads = [0] * self._num_replicas

        while self._request_queue:
            request = self._request_queue.pop(0)
            remaining_tokens = request.total_tokens - request.num_processed_tokens
            
            # Find the replica with the least load
            replica_id = min(range(self._num_replicas), key=lambda i: replica_loads[i])
            replica_loads[replica_id] += remaining_tokens
            request_mapping.append((replica_id, request))

        return request_mapping
