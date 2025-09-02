from math import ceil
from typing import List

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


class SJFReplicaScheduler(BaseReplicaScheduler):
    """
    Shortest Job First (SJF) Replica Scheduler.
    
    This scheduler orders requests by their total token count (job size) 
    before processing them. Shorter jobs get higher priority.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._preempted_requests: List[Request] = []
        self._num_running_batches = 0
        # Use a simple batching approach similar to VLLM
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages
        self._watermark_blocks = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks
        )

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        for request in batch.requests:
            if request.completed:
                self.free(request.id)
            else:
                self._preempted_requests.append(request)

    def _can_allocate_request(self, request: Request) -> bool:
        if request.id not in self._allocation_map:
            # new request
            num_required_blocks = ceil(
                (request.num_prefill_tokens) / self._config.block_size
            )
            return (
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks
            )

        # need at least one block to be available for decode
        return self._config.num_blocks - self._num_allocated_blocks >= 1

    def _allocate_request(self, request: Request) -> None:
        if request.id not in self._allocation_map:
            # new request - allocate blocks for prefill tokens
            num_required_blocks = ceil(
                (request.num_prefill_tokens) / self._config.block_size
            )
            self.allocate(request.id, num_required_blocks)
        else:
            # continuing request - allocate one more block for decode
            self.allocate(request.id, 1)

    def _sort_requests_by_job_size(self) -> None:
        """Sort request queue by total token count (job size) - shortest first."""
        self._request_queue.sort(key=lambda request: request.total_tokens)

    def _get_next_batch(self) -> Batch:
        # Always sort requests by job size before processing
        self._sort_requests_by_job_size()
        
        # First, try to add preempted requests back
        requests_to_schedule = []
        for request in self._preempted_requests[:]:
            if self._can_allocate_request(request):
                self._allocate_request(request)
                requests_to_schedule.append(request)
                self._preempted_requests.remove(request)

                if len(requests_to_schedule) >= self._max_micro_batch_size:
                    break

        # Then add new requests in SJF order
        while (
            len(requests_to_schedule) < self._max_micro_batch_size
            and self._request_queue
        ):
            request = self._request_queue[0]  # Get the shortest job first
            if self._can_allocate_request(request):
                self._allocate_request(request)
                requests_to_schedule.append(request)
                self._request_queue.pop(0)
            else:
                # Can't allocate memory for this request, stop trying
                break

        if not requests_to_schedule:
            return None

        # Create batch with next number of tokens for each request
        num_tokens = [
            self._get_request_next_num_tokens(request)
            for request in requests_to_schedule
        ]
        
        print(f"Scheduling {len(requests_to_schedule)} requests with total tokens: {num_tokens}")

        return Batch(
            self._replica_id,
            requests_to_schedule,
            num_tokens,
        )

    def estimate_ttft_ms_if_enqueued_now(self, request: "Request") -> float:
        """
        Estimate TTFT if this request were added to the queue now.
        For SJF, we need to consider where this request would be positioned
        in the sorted queue based on its job size.
        """
        # Simple estimation: assume some base processing time
        # In practice, this would use more sophisticated models
        base_ttft_ms = 50.0  # Base TTFT estimate
        
        # Count how many requests with smaller job sizes are ahead
        smaller_jobs = sum(1 for req in self._request_queue 
                          if req.total_tokens < request.total_tokens)
        
        # Add estimated delay based on queue position  
        queue_delay_ms = smaller_jobs * 20.0  # Rough estimate of 20ms per request ahead
        
        return base_ttft_ms + queue_delay_ms
