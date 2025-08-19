from math import ceil
from typing import List

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


class SRPTReplicaScheduler(BaseReplicaScheduler):
    """
    Shortest Remaining Processing Time (SRPT) Replica Scheduler.
    
    This scheduler prioritizes requests based on their remaining work.
    Requests with less remaining tokens to process get higher priority.
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

    def _get_remaining_tokens(self, request: Request) -> int:
        """Calculate remaining tokens for a request (total - processed)."""
        if request.stage == "prefill":
            # During prefill, no tokens have been generated yet
            return request.total_tokens
        else:
            # During decode, some tokens have been generated
            generated_tokens = len(request.generated_tokens) if request.generated_tokens else 0
            return request.num_decode_tokens - generated_tokens

    def _sort_requests_by_remaining_time(self) -> None:
        """Sort request queue by remaining processing time - shortest remaining first."""
        self._request_queue.sort(key=lambda request: self._get_remaining_tokens(request))

    def _get_next_batch(self) -> Batch:
        # Always sort requests by remaining processing time before processing
        self._sort_requests_by_remaining_time()
        
        # First, try to add preempted requests back (they also get re-sorted)
        requests_to_schedule = []
        for request in self._preempted_requests[:]:
            if self._can_allocate_request(request):
                self._allocate_request(request)
                requests_to_schedule.append(request)
                self._preempted_requests.remove(request)

                if len(requests_to_schedule) >= self._max_micro_batch_size:
                    break

        # Sort preempted requests by remaining time as well
        if self._preempted_requests:
            self._preempted_requests.sort(key=lambda request: self._get_remaining_tokens(request))

        # Then add new requests in SRPT order
        while (
            len(requests_to_schedule) < self._max_micro_batch_size
            and self._request_queue
        ):
            request = self._request_queue[0]  # Get the request with shortest remaining time
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

        return Batch(
            self._replica_id,
            requests_to_schedule,
            num_tokens,
        )

    def estimate_ttft_ms_if_enqueued_now(self, request: "Request") -> float:
        """
        Estimate TTFT if this request were added to the queue now.
        For SRPT, we need to consider where this request would be positioned
        based on its remaining processing time.
        """
        # Simple estimation: assume some base processing time
        base_ttft_ms = 50.0  # Base TTFT estimate
        
        # Calculate this request's remaining tokens (it's new, so all tokens remain)
        request_remaining = request.total_tokens
        
        # Count how many requests have less remaining work
        requests_with_less_work = sum(
            1 for req in self._request_queue 
            if self._get_remaining_tokens(req) < request_remaining
        )
        
        # Add estimated delay based on queue position  
        queue_delay_ms = requests_with_less_work * 20.0  # Rough estimate of 20ms per request ahead
        
        return base_ttft_ms + queue_delay_ms
