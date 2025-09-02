from math import ceil
from typing import List, Tuple
import heapq

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


class SJFReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._preempted_requests: List[Request] = []
        self._num_running_batches = 0
        # For vLLM and its derivatives, we only need to set a loose max batch size
        # Memory requirements are handled explicitly by the scheduler
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

        # vllm requires at least one block to be available
        return self._config.num_blocks - self._num_allocated_blocks >= 1

    def _allocate_request(self, request: Request) -> None:
        if request.id not in self._allocation_map:
            # new request
            num_required_blocks = ceil(
                (request.num_prefill_tokens) / self._config.block_size
            )
            self.allocate(request.id, num_required_blocks)
            return

        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)
        assert (
            num_tokens_required == 0 or num_tokens_required == 1
        ), f"num_tokens_required: {num_tokens_required}"

        if num_tokens_required == 0:
            return

        self.allocate(request.id, 1)

    def _calculate_request_weight(self, request: Request) -> float:
        """
        Calculate the computational weight of a request.
        Weight = prompt_len * prompt_len - prefix_hit * prefix_hit - computed_prompt_len * computed_prompt_len
        
        For simplicity, we'll use:
        - prompt_len = num_prefill_tokens (for new requests) or remaining decode tokens
        - prefix_hit = 0 (assuming no prefix caching for now)
        - computed_prompt_len = num_processed_tokens
        """
        if request.id not in self._allocation_map:
            # New request: weight is based on prefill computation
            prompt_len = request.num_prefill_tokens
            prefix_hit = 0  # No prefix caching assumed
            computed_prompt_len = 0  # Nothing computed yet
        else:
            # Continuing request: weight is based on decode computation
            prompt_len = request.num_prefill_tokens + request.num_decode_tokens
            prefix_hit = 0  # No prefix caching assumed  
            computed_prompt_len = request.num_processed_tokens
        
        # Computational cost is quadratic in sequence length for attention
        weight = prompt_len * prompt_len - prefix_hit * prefix_hit - computed_prompt_len * computed_prompt_len
        return max(weight, 1.0)  # Ensure positive weight

    def _calculate_request_value(self, request: Request) -> float:
        """
        Calculate the value of a request.
        Value = prompt_len (prioritize shorter prompts to enable larger batches)
        
        For SJF, we want to prioritize shorter jobs, so we use negative prompt length
        or inverse of prompt length to make shorter prompts have higher value.
        """
        if request.id not in self._allocation_map:
            # New request
            prompt_len = request.num_prefill_tokens
        else:
            # Continuing request - use remaining tokens
            remaining_tokens = request.num_decode_tokens - (request.num_processed_tokens - request.num_prefill_tokens)
            prompt_len = max(remaining_tokens, 1)
        
        # Higher value for shorter prompts (SJF principle)
        return 1.0 / prompt_len

    def _solve_knapsack(self, candidates: List[Request], compute_budget: float) -> List[Request]:
        """
        Solve the knapsack problem to select optimal batch composition.
        
        Args:
            candidates: List of candidate requests (both queued and preempted)
            compute_budget: Available compute budget
            
        Returns:
            List of selected requests that maximize value within budget
        """
        if not candidates:
            return []
        
        # Calculate weight and value for each candidate
        items = []
        for i, request in enumerate(candidates):
            weight = self._calculate_request_weight(request)
            value = self._calculate_request_value(request)
            items.append((i, weight, value, request))
        
        # Sort by value/weight ratio (greedy approximation for efficiency)
        items.sort(key=lambda x: x[2] / x[1], reverse=True)
        
        selected_requests = []
        total_weight = 0.0
        
        for idx, weight, value, request in items:
            # Check if we can allocate this request
            if not self._can_allocate_request(request):
                continue
                
            # Check if adding this request exceeds compute budget
            if total_weight + weight <= compute_budget:
                # Check batch size constraints
                if len(selected_requests) >= self._max_micro_batch_size:
                    break
                    
                # Check token constraints
                next_num_tokens = self._get_request_next_num_tokens(request)
                current_max_tokens = max([self._get_request_next_num_tokens(r) for r in selected_requests] + [0])
                new_max_tokens = max(current_max_tokens, next_num_tokens)
                new_batch_tokens = (len(selected_requests) + 1) * new_max_tokens
                
                if new_batch_tokens > self._config.max_tokens_in_batch:
                    continue
                    
                if len(self._allocation_map) + len(selected_requests) >= self._config.batch_size_cap:
                    break
                
                selected_requests.append(request)
                total_weight += weight
            
        return selected_requests

    def _get_compute_budget(self) -> float:
        """
        Calculate available compute budget.
        This could be based on available GPU memory, time constraints, etc.
        For now, use a simple heuristic based on max tokens in batch.
        """
        # Use max_tokens_in_batch as a proxy for compute budget
        # Scale it to account for quadratic attention cost
        base_budget = self._config.max_tokens_in_batch
        
        # Account for current memory usage
        memory_pressure = self._num_allocated_blocks / self._config.num_blocks
        adjusted_budget = base_budget * (1.0 - memory_pressure * 0.5)
        
        return max(adjusted_budget, 1000.0)  # Minimum budget

    def _get_next_batch(self) -> Batch:
        """
        Get next batch using knapsack optimization.
        """
        # Combine queued and preempted requests as candidates
        candidates = list(self._request_queue) + list(self._preempted_requests)
        # Filter out completed requests
        candidates = [r for r in candidates if not r._completed]
        print(f"Knapsack candidates: {len(candidates)} requests: {[r.id for r in candidates]}")
        
        if not candidates:
            return None
        
        # Calculate compute budget
        compute_budget = self._get_compute_budget()
        print(f"Compute budget: {compute_budget:.1f}")
        
        # Solve knapsack problem to select optimal requests
        selected_requests = self._solve_knapsack(candidates, compute_budget)
        print(f"Knapsack selected: {len(selected_requests)} requests: {[r.id for r in selected_requests]}")
        
        if not selected_requests:
            return None
        
        # Remove selected requests from their respective queues
        for request in selected_requests:
            if request in self._request_queue:
                self._request_queue.remove(request)
            if request in self._preempted_requests:
                self._preempted_requests.remove(request)
        
        # Allocate resources and prepare batch
        batch_requests = []
        num_tokens = []
        
        for request in selected_requests:
            try:
                self._allocate_request(request)
                next_num_tokens = self._get_request_next_num_tokens(request)
                batch_requests.append(request)
                num_tokens.append(next_num_tokens)
            except Exception as e:
                # If allocation fails, put request back in appropriate queue
                if request.id not in self._allocation_map:
                    self._request_queue.insert(0, request)
                else:
                    self._preempted_requests.insert(0, request)
                print(f"Failed to allocate request {request.id}: {e}")
        
        if not batch_requests:
            return None
        
        print(f"Knapsack batch: {len(batch_requests)} requests, "
              f"total weight: {sum(self._calculate_request_weight(r) for r in batch_requests):.1f}, "
              f"total value: {sum(self._calculate_request_value(r) for r in batch_requests):.3f}")
        
        
        return Batch(self._replica_id, batch_requests, num_tokens)

    def estimate_ttft_ms_if_enqueued_now(self, request: "Request") -> float:
        """
        Estimate TTFT if this request were added to the queue now.
        For knapsack-based SJF, this depends on the value/weight ratio.
        """
        base_ttft_ms = 50.0
        
        # Calculate this request's value/weight ratio
        weight = self._calculate_request_weight(request)
        value = self._calculate_request_value(request)
        ratio = value / weight
        
        # Count requests with better ratios (higher priority)
        better_requests = 0
        for req in self._request_queue:
            req_weight = self._calculate_request_weight(req)
            req_value = self._calculate_request_value(req)
            req_ratio = req_value / req_weight
            if req_ratio > ratio:
                better_requests += 1
        
        # Estimate delay based on queue position
        queue_delay_ms = better_requests * 15.0
        
        return base_ttft_ms + queue_delay_ms
