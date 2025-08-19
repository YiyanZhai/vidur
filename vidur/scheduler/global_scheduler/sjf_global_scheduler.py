from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class SJFGlobalScheduler(BaseGlobalScheduler):
    """
    Shortest Job First (SJF) global scheduler.
    Schedules requests in order of their total token count (job size),
    with shorter jobs having higher priority.
    """

    def schedule(self) -> List[Tuple[int, Request]]:
        # Sort requests by arrival time first (base behavior)
        self.sort_requests()

        # Sort by total tokens (job size) - shortest jobs first
        self._request_queue.sort(key=lambda request: request.total_tokens)

        request_mapping = []
        replica_loads = [0] * self._num_replicas

        while self._request_queue:
            request = self._request_queue.pop(0)
            # Find the replica with the least load
            replica_id = min(range(self._num_replicas), key=lambda i: replica_loads[i])
            replica_loads[replica_id] += request.total_tokens
            request_mapping.append((replica_id, request))

        return request_mapping
