from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class SJFGlobalScheduler(BaseGlobalScheduler):
    """
    Shortest Job First (SJF) global scheduler.
    Schedules requests in order of their total token count (job size),
    with shorter jobs having higher priority. Uses load balancing based on
    pending tokens at each replica.
    """

    def schedule(self) -> List[Tuple[int, Request]]:
        # Sort requests by arrival time first (base behavior)
        self.sort_requests()

        # Sort by total tokens (job size) - shortest jobs first
        self._request_queue.sort(key=lambda request: request.total_tokens)

        request_mapping = []
        # Keep a map of replica_id -> pending tokens
        # This tracks the current load at each replica scheduler
        pending_tokens_map = {
            replica_scheduler.replica_id: replica_scheduler.num_pending_tokens
            for replica_scheduler in self._replica_schedulers.values()
        }

        while self._request_queue:
            request = self._request_queue.pop(0)
            # Find the replica with the least pending tokens (least work)
            replica_id = min(pending_tokens_map.items(), key=lambda x: x[1])[0]
            # Update the pending tokens for this replica
            pending_tokens_map[replica_id] += request.total_tokens
            request_mapping.append((replica_id, request))    
        
        return request_mapping
