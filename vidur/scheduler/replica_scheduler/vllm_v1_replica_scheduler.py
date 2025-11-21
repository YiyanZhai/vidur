from collections import deque
from typing import Deque, Dict, List

from vidur.entities.batch import Batch, Request
from vidur.kv_cache.replica_kv_cache_manager import ReplicaKVCacheManager
from vidur.logger import init_logger
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.outsourcing import (
    APICostCalculator,
    CandidateSelector,
    KnapsackSolver,
    RequestTracker,
    TTFTEstimateTracker,
    TTFTViolationDetector,
)
from vidur.scheduler.replica_scheduler.replica_scheduler_output import (
    ReplicaSchedulerOutput,
)
from vidur.types.request_queue_type import RequestQueueType

logger = init_logger(__name__)


class VLLMV1ReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert (
            self._waiting_queue._config.get_type() == RequestQueueType.FCFS
        ), "VLLM_v1 scheduler only supports FCFS request queues"
        assert (
            self._num_stages == 1
        ), "VLLM_v1 scheduler doesn't support pipeline parallelism"

        # Scheduling constraints
        self._max_batch_size = self._config.batch_size_cap
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages

        # Create the KV Cache manager
        self._kv_cache_manager = ReplicaKVCacheManager(
            block_size=self._cache_config.block_size,
            num_gpu_blocks=self._cache_config.num_blocks,
            enable_caching=self._cache_config.enable_prefix_caching,
            caching_hash_algo=self._cache_config.prefix_caching_hash_algo,
            num_preallocate_tokens=self._cache_config.num_preallocate_tokens,
        )

        # req_id -> Request
        self._requests: Dict[str, Request] = {}
        # self._waiting_queue has been initialized in the parent class
        self._running: List[Request] = []
        # The requests that have been scheduled and are being executed
        # by the executor.
        self.scheduled_req_ids: set[str] = set()
        
        # Outsourcing state
        self._outsourced_req_ids: set[str] = set()
        
        # Initialize outsourcing configuration
        self._init_outsourcing_config()
    
    # ==================== Configuration & Initialization ====================
        
    def _init_outsourcing_config(self):
        """Initialize outsourcing-related configuration parameters."""
        # Throughput estimates (tokens/sec)
        # Use dynamic estimation if available, otherwise fall back to config/default
        use_dynamic_throughput = getattr(
            self._config, 'use_dynamic_prefill_throughput', True
        )
        if use_dynamic_throughput:
            self._prefill_throughput = None  # Will be computed dynamically
        else:
            self._prefill_throughput = getattr(
                self._config, 'prefill_tokens_per_sec', 1000
            )
        
        # Weight ratio for decode vs prefill in knapsack
        self._decode_weight_ratio = getattr(
            self._config, 'decode_weight_ratio', 0.6
        )
        
        # Budget horizon (iterations to look ahead)
        self._budget_horizon = getattr(
            self._config, 'budget_horizon_iterations', 2
        )
        
        # Debug logging flag
        self._debug_outsourcing = getattr(
            self._config, 'debug_outsourcing', False
        )
        
        # Initialize outsourcing components
        self._cost_calculator = APICostCalculator(
            input_price_per_million=getattr(self._config, 'input_price_per_million', 1.25),
            output_price_per_million=getattr(self._config, 'output_price_per_million', 10.00),
        )
        
        self._request_tracker = RequestTracker(
            replica_id=self._replica_id,
            cost_calculator=self._cost_calculator.calculate_cost,
        )
        
        self._candidate_selector = CandidateSelector()
        
        strategy = getattr(self._config, 'knapsack_strategy', 'dp_scaled')
        self._knapsack_solver = KnapsackSolver(strategy=strategy)
        
        # Initialize TTFT estimate tracker
        self._ttft_tracker = TTFTEstimateTracker()
        
        violation_mode = getattr(self._config, 'ttft_violation_mode', 'all')
        self._violation_detector = TTFTViolationDetector(
            mode=violation_mode,
            prefill_throughput=self._get_prefill_throughput_estimate(),
            max_micro_batch_size=self._max_micro_batch_size,
            ttft_tracker=self._ttft_tracker,  # Pass tracker to violation detector
        )
    
    def _get_prefill_throughput_estimate(self, chunk_size: int = None) -> float:
        """
        Estimate prefill throughput (tokens/sec) using the execution time predictor.
        
        This method creates a synthetic batch to query the execution time predictor
        and calculates an approximate throughput based on predicted execution time.
        
        Args:
            chunk_size: Size of prefill chunk to use for estimation. 
                       Defaults to configured chunk_size.
        
        Returns:
            Estimated prefill throughput in tokens per second.
        """
        # If static throughput is configured, use it
        if self._prefill_throughput is not None:
            return self._prefill_throughput
        
        # Use chunk_size for estimation
        if chunk_size is None:
            chunk_size = self._config.chunk_size
        
        # Create a synthetic request for throughput estimation
        # We'll use a simple case: single request, no KV cache
        from vidur.entities import Request as RequestEntity
        
        # Create a dummy request (we only need it for the batch structure)
        dummy_request = RequestEntity(
            arrived_at=0.0,
            num_prefill_tokens=chunk_size,
            num_decode_tokens=1,
            block_hash_ids=None,
            block_size=15
        )
        
        # Create a synthetic batch with the chunk size
        synthetic_batch = Batch(
            replica_id=self._replica_id,
            requests=[dummy_request],
            num_tokens=[chunk_size],
        )
        
        # Get execution time prediction from the predictor
        # Pipeline stage 0 since we don't support pipeline parallelism
        exec_time = self._execution_time_predictor.get_batch_execution_time(
            synthetic_batch, pipeline_stage=0
        )
        
        # Calculate total time for prefill in milliseconds
        # Use model_time_ms if available (more accurate), otherwise sum components
        if hasattr(exec_time, 'model_time_ms') and exec_time.model_time_ms is not None:
            total_time_ms = float(exec_time.model_time_ms)
        else:
            # Sum up the relevant prefill execution time components
            per_layer_ms = (
                exec_time.attention_prefill_execution_time +
                exec_time.attention_layer_pre_proj_execution_time +
                exec_time.attention_layer_post_proj_execution_time +
                exec_time.attention_rope_execution_time +
                exec_time.attention_kv_cache_save_execution_time +
                exec_time.mlp_up_proj_time +
                exec_time.mlp_down_proj_time +
                exec_time.mlp_act_time +
                exec_time.attn_norm_time +
                exec_time.mlp_norm_time +
                exec_time.add_time
            )
            
            total_time_ms = per_layer_ms * self._replica_config.model_config.num_layers
            
            # Add communication overhead if present
            if exec_time.attention_all_reduce_time > 0:
                total_time_ms += exec_time.attention_all_reduce_time
            if exec_time.pipeline_parallel_communication_time > 0:
                total_time_ms += exec_time.pipeline_parallel_communication_time
        
        # Convert to seconds
        total_time_sec = total_time_ms / 1000.0
        
        # Calculate throughput: tokens / time
        if total_time_sec > 0:
            throughput = chunk_size / total_time_sec
        else:
            # Fallback to default if prediction fails
            logger.warning(f"[Replica {self._replica_id}] Predicted execution time is zero, "
                         f"using fallback throughput of 1000 tokens/sec")
            throughput = 1000.0
        
        if self._debug_outsourcing:
            logger.info(f"[Replica {self._replica_id}] Estimated prefill throughput: "
                       f"{throughput:.1f} tokens/sec (chunk_size={chunk_size}, "
                       f"exec_time={total_time_ms:.2f}ms)")
        
        return throughput
        
    # ==================== Outsourcing Orchestration ====================
    
    def _maybe_outsource_before_schedule(self, current_time: float) -> None:
        """
        Main outsourcing decision hook called before each scheduling step.
        Iteratively outsources one request at a time until TTFT violations are resolved.
        """
        if not len(self._waiting_queue) and not self._running:
            return

        # Iteratively outsource one request at a time until no violations
        iteration = 0
        max_iterations = 100  # Safety limit to prevent infinite loops
        
        while iteration < max_iterations:
            # Check for TTFT violations
            waiting_list = self._waiting_queue.to_list() if hasattr(self._waiting_queue, "to_list") else list(self._waiting_queue)
            if not self._violation_detector.check_violations(
                waiting_list, self.get_cached_prefill_length, current_time
            ):
                # No violations detected, we're done
                if iteration > 0 and self._debug_outsourcing:
                    logger.info(f"[Replica {self._replica_id}] TTFT violations resolved after {iteration} outsourcing iteration(s)")
                return
            
            if iteration == 0 and self._debug_outsourcing:
                logger.info(f"[Replica {self._replica_id}] TTFT violation detected at t={current_time:.2f}")
            
            # Collect candidates
            candidates = self._candidate_selector.collect_candidates(
                waiting_requests=waiting_list,
                running_requests=self._running,
                outsourced_req_ids=self._outsourced_req_ids,
                scheduled_req_ids=self.scheduled_req_ids,
            )
            
            if not candidates:
                if self._debug_outsourcing:
                    logger.info(f"[Replica {self._replica_id}] No more outsourcing candidates, stopping at iteration {iteration}")
                return

            # Build knapsack items
            items = [self._knapsack_item_for(r) for r in candidates]
            
            # Calculate total weight needed to keep all candidates local
            total_weight = sum(item["weight"] for item in items)
            
            # Set budget to total_weight - 1 to force outsourcing of at least one request
            # This ensures we select all but one request to keep local
            budget = max(1, total_weight - 1)
            
            # Solve knapsack - this will select requests to KEEP local
            keep_ids, outsource_ids = self._knapsack_solver.solve(items, budget)
            
            # If knapsack couldn't outsource anything (shouldn't happen with budget = total - 1)
            # fall back to outsourcing the lowest value request
            if not outsource_ids:
                # Sort by value (lowest first) and outsource the cheapest one
                sorted_items = sorted(items, key=lambda x: x["value"])
                outsource_ids = [sorted_items[0]["id"]]
                if self._debug_outsourcing:
                    logger.info(f"[Replica {self._replica_id}] Knapsack didn't outsource, manually selecting lowest-value request")
            
            # Outsource only ONE request (the first one selected)
            # This is more conservative than outsourcing all at once
            single_outsource = [outsource_ids[0]] if outsource_ids else []
            
            if single_outsource:
                if self._debug_outsourcing:
                    logger.info(f"[Replica {self._replica_id}] Iteration {iteration + 1}: Outsourcing 1 request: {single_outsource[0]}")
                self._apply_outsourcing(single_outsource, current_time)
                iteration += 1
            else:
                # No request to outsource, break
                if self._debug_outsourcing:
                    logger.info(f"[Replica {self._replica_id}] No request selected for outsourcing at iteration {iteration}")
                return
        
        # Safety limit reached
        if self._debug_outsourcing:
            logger.warning(f"[Replica {self._replica_id}] Reached max outsourcing iterations ({max_iterations}), violations may still exist")

    # ==================== Knapsack Item Construction ====================
    
    def _knapsack_item_for(self, r: Request) -> dict:
        """
        Convert a request into a knapsack item.
        Weight: remaining work (prefill + weighted decode)
        Value: cost savings from keeping local (API cost avoided)
        """
        cached = self.get_cached_prefill_length(r)
        processed = r.num_processed_tokens
        prefill_done = max(processed, cached)
        rem_prefill = max(0, r.num_prefill_tokens - prefill_done)
        decode_done = max(0, processed - r.num_prefill_tokens)
        rem_decode = max(0, r.num_decode_tokens - decode_done)
        
        # Weight = remaining work (normalized by decode ratio)
        # Weight = remaining FLOPs needed
        # Prefill FLOPs: 2 * n * d * (d_ff + d_model)
        # Decode FLOPs: 2 * d * (d_ff + d_model) per token
        # Simplified: prefill_flops ≈ 2 * n * d^2, decode_flops ≈ 2 * d^2
        # Ratio: prefill_flops / decode_flops ≈ n (sequence length)
        
        # Get model dimensions from config
        d_model = self._replica_config.model_config.embedding_dim
        d_ff = self._replica_config.model_config.mlp_hidden_dim
        num_layers = self._replica_config.model_config.num_layers
        
        # Calculate FLOPs per token (simplified formula)
        # Forward pass FLOPs ≈ 2 * num_layers * (4 * d_model^2 + 2 * d_model * d_ff)
        flops_per_token = 2 * num_layers * (4 * d_model * d_model + 2 * d_model * d_ff)
        
        # Prefill is more expensive per token due to attention computation
        # Attention FLOPs scale with sequence length: O(n^2 * d)
        # For simplicity, use average sequence position for prefill
        avg_seq_len = (r.num_prefill_tokens + 1) / 2
        prefill_flops = rem_prefill * flops_per_token * avg_seq_len
        decode_flops = rem_decode * flops_per_token
        
        weight = int(prefill_flops + self._decode_weight_ratio * decode_flops)
        
        # Value = cost savings from NOT outsourcing (API cost avoided)
        api_cost = self._cost_calculator.calculate_cost(rem_prefill, rem_decode)
        value = int(api_cost * 1000)  # Scale to avoid float issues in DP
        
        return {"id": r.id, "weight": max(1, weight), "value": max(1, value)}

    def _local_prefill_budget_horizon(self) -> int:
        """Calculate the local budget for prefill work over the next horizon iterations."""
        return self._budget_horizon * self._max_micro_batch_size * self._config.chunk_size
    
    # ==================== Request Removal & Tracking ====================

    def _apply_outsourcing(self, outsource_ids: list[str], current_time: float) -> None:
        """
        Remove outsourced requests from waiting queue and running list.
        Optimized with O(n) set lookup instead of nested loops.
        """
        if not outsource_ids:
            return
        
        outsource_set = set(outsource_ids)
        waiting_count = 0
        running_count = 0
        
        # 1) Remove from waiting queue and track
        snapshot = self._waiting_queue.to_list() if hasattr(self._waiting_queue, "to_list") else []
        kept = []
        for r in snapshot:
            if r.id in outsource_set:
                self._outsourced_req_ids.add(r.id)
                self._track_outsourced_request(r, was_running=False, current_time=current_time)
                # Clear TTFT estimate since request is outsourced
                self._ttft_tracker.clear_estimate(r.id)
                waiting_count += 1
            else:
                kept.append(r)
        
        self._waiting_queue.clear()
        for r in kept:
            self._waiting_queue.push(r)

        # 2) Preempt running prefill requests if selected
        new_running = []
        for r in self._running:
            if r.id in outsource_set:
                # Free KV, restart to normalize internal counters, then drop
                self._kv_cache_manager.free(r)
                r.restart()
                self._kv_cache_manager.free_block_hashes(r)
                self.scheduled_req_ids.discard(r.id)
                self._requests.pop(r.id, None)
                self._outsourced_req_ids.add(r.id)
                self._track_outsourced_request(r, was_running=True, current_time=current_time)
                # Clear TTFT estimate since request is outsourced
                self._ttft_tracker.clear_estimate(r.id)
                running_count += 1
            else:
                new_running.append(r)
        self._running = new_running
        
        if self._debug_outsourcing:
            logger.info(f"[Replica {self._replica_id}] Outsourced {waiting_count} waiting + {running_count} running requests")
    
    def _track_outsourced_request(self, request: Request, was_running: bool, current_time: float) -> None:
        """Track details of an outsourced request using the RequestTracker."""
        self._request_tracker.track_outsourced_request(request, was_running, current_time)
    
    # ==================== Public API for Metrics Collection ====================
    
    def get_outsourced_request_details(self) -> List[dict]:
        """Return the list of outsourced request details."""
        return self._request_tracker.get_outsourced_request_details()
    
    def get_outsourcing_statistics(self) -> dict:
        """Calculate and return outsourcing statistics."""
        return self._request_tracker.get_outsourcing_statistics()
        
    @property
    def memory_usage_percent(self) -> float:
        return self._kv_cache_manager.usage * 100

    def get_cached_prefill_length(self, request: Request) -> int:
        _, num_computed_tokens = self._kv_cache_manager.get_computed_blocks(request)
        return num_computed_tokens

    def add_request(self, request: Request):
        request.assign_replica(self._replica_id)
        self._waiting_queue.push(request)
        self._requests[request.id] = request

    def _get_request_next_num_tokens(self, request: Request, token_budget: int) -> int:
        assert not request.completed

        # Calculate `next_num_tokens`
        if request.is_prefill_complete:
            next_num_tokens = 1
        else:
            next_num_tokens = request.num_prefill_tokens - request.num_processed_tokens
        # Pass through the token budget
        next_num_tokens = min(next_num_tokens, token_budget)
        # No negative answer
        next_num_tokens = max(0, next_num_tokens)
        return next_num_tokens

    def _get_next_batch(self, current_time: float) -> ReplicaSchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_reqs: List[Request] = []
        preempted_reqs: List[Request] = []
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self._config.chunk_size

        # First, schedule the RUNNING requests
        req_index = 0
        while req_index < len(self._running) and token_budget > 0:
            request: Request = self._running[req_index]
            if request.id in self.scheduled_req_ids:
                req_index += 1
                continue

            # Calculate compute to do for the request
            num_new_tokens = self._get_request_next_num_tokens(request, token_budget)
            assert (
                num_new_tokens > 0
            ), "num_new_tokens should be as token_budget > 0 and request is incomplete"

            # Try to allocate memory for the request
            while True:
                new_blocks = self._kv_cache_manager.allocate_slots(
                    request, num_new_tokens
                )
                if new_blocks is None:
                    # print(f"Cannot schedule request {request.id} due to memory constraints.")
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    preempted_req: Request = self._running.pop()  # from last
                    self._kv_cache_manager.free(preempted_req)
                    preempted_req.restart()
                    self._waiting_queue.push(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_reqs.append(request)
            self.scheduled_req_ids.add(request.id)
            num_scheduled_tokens[request.id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

        # Use a temporary deque to collect requests that need to be skipped
        # and put back at the head of the waiting queue later
        skipped_waiting_requests: Deque[Request] = deque()

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while len(self._waiting_queue) and token_budget > 0:
                if len(self._running) >= self._max_micro_batch_size:
                    break

                request = self._waiting_queue.peek()

                # Get already-cached tokens. `computed` means `cached` here.
                computed_blocks, num_computed_tokens = (
                    self._kv_cache_manager.get_computed_blocks(request)
                )
                # Number of tokens to be scheduled.
                # Using `request.num_prefill_tokens` is fine even for restarted requests
                # because done decode tokens have been added to prefill tokens.
                num_new_tokens = request.num_prefill_tokens - num_computed_tokens
                if num_new_tokens < 0:
                    # This can happen when block_hash_ids from trace don't match actual prefill tokens,
                    # often due to session_id collisions or trace generation bugs.
                    # Treat as no cache hit and process all prefill tokens.
                    print(
                        f"WARNING: Request {request.id} has more cached tokens ({num_computed_tokens}) "
                        f"than prefill tokens ({request.num_prefill_tokens}). "
                        f"Possible session_id collision in trace. Ignoring cache."
                    )
                    computed_blocks = []
                    num_computed_tokens = 0
                    num_new_tokens = request.num_prefill_tokens
                elif num_new_tokens == 0:
                    # This happens when prompt length is divisible by the block
                    # size and all blocks are cached. Now we force to recompute
                    # the last block. Note that we have to re-compute an entire
                    # block because allocate_slots() assumes num_computed_tokens
                    # is always a multiple of the block size. This limitation
                    # can potentially be removed in the future to slightly
                    # improve the performance.
                    num_computed_tokens -= self._cache_config.block_size
                    num_new_tokens = self._cache_config.block_size
                    computed_blocks.pop()
                num_new_tokens = min(num_new_tokens, token_budget)
                assert (
                    num_new_tokens > 0
                ), f"num_new_tokens should be greater than 0 but got {num_new_tokens}"

                new_blocks = self._kv_cache_manager.allocate_slots(
                    request, num_new_tokens, computed_blocks
                )
                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                self._waiting_queue.pop()
                req_index += 1
                self._running.append(request)
                self.scheduled_req_ids.add(request.id)
                scheduled_reqs.append(request)
                assert not request.scheduled
                num_scheduled_tokens[request.id] = num_new_tokens
                token_budget -= num_new_tokens
                # Update the number of processed tokens for the request
                request.on_cache_hit(num_computed_tokens)

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self._waiting_queue.extend(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self._config.chunk_size
        assert token_budget >= 0
        assert len(self._running) <= self._max_micro_batch_size
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_reqs) <= len(self._running)

        # print(f"Replica {self._replica_id} scheduling: running {[r.id for r in self._running]}, waiting {[r.id for r in self._waiting_queue.to_list()] if self._waiting_queue else []}, scheduled {[(request.id, num_scheduled_tokens[request.id]) for request in scheduled_reqs]}, outsourced {len(self._outsourced_req_ids)}, mem {self.memory_usage_percent:.1f}%")

        scheduler_output = ReplicaSchedulerOutput(
            (
                Batch(
                    self._replica_id,
                    scheduled_reqs,
                    [num_scheduled_tokens[request.id] for request in scheduled_reqs],
                )
                if scheduled_reqs
                else None
            ),
            [],
        )
        # TODO(nitin): Immediately updating num_processed_tokens for the request is important for
        #  sequence pipeline parallelism and multi-step scheduling.
        # However, this is not done here to protect the invariant that num_processed_tokens is updated only after batch end.
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        # for req_id, num_scheduled_token in num_scheduled_tokens.items():
        #     self._requests[req_id].num_processed_tokens += num_scheduled_token

        self.finished_req_ids = set()
        return scheduler_output

    def is_empty(self) -> bool:
        return len(self._waiting_queue) + len(self._running) == 0

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1
        new_running: List[Request] = []

        # NOTE(woosuk): As len(self.running) can be up to 1K or more, the below
        # loop can be a performance bottleneck. We should do our best to avoid
        # expensive operations inside the loop.
        for request in self._running:
            req_id = request.id
            num_tokens_scheduled = batch.num_tokens_dict.get(req_id, 0)
            if num_tokens_scheduled == 0:
                # The request was not scheduled in this step.
                new_running.append(request)
                continue
            elif request.completed:
                self._free_request(request)
            else:
                new_running.append(request)
            self.scheduled_req_ids.remove(req_id)
        self._running = new_running

    def _free_request(self, request: Request) -> None:
        assert request.completed
        self._kv_cache_manager.free(request)
        self._kv_cache_manager.free_block_hashes(request)
        del self._requests[request.id]
    
    def save_ttft_estimates(self, output_dir: str) -> None:
        """
        Save tracked TTFT estimates to a CSV file.
        
        Args:
            output_dir: Directory where to save the estimates file
        """
        import os
        filepath = os.path.join(output_dir, f"ttft_estimates_replica_{self._replica_id}.csv")
        self._ttft_tracker.save_to_csv(filepath)
        
        # Also log summary stats
        # stats = self._ttft_tracker.get_summary_stats()
        # if stats:
        #     logger.info(f"[Replica {self._replica_id}] TTFT Estimate Summary: {stats}")

