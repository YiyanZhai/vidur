from collections import deque
from typing import Deque, Dict, List

from vidur.entities.batch import Batch, Request
from vidur.kv_cache.replica_kv_cache_manager import ReplicaKVCacheManager
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.replica_scheduler_output import (
    ReplicaSchedulerOutput,
)
from vidur.types.request_queue_type import RequestQueueType


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
        self._outsourced_req_ids: set[str] = set()

    # ---- Outsourcing hook ----
    def _maybe_outsource_before_schedule(self, current_time: float) -> None:
        if not len(self._waiting_queue) and not self._running:
            return

        print(f"Considering outsourcing: {self._waiting_queue.to_list() if hasattr(self._waiting_queue, 'to_list') else list(self._waiting_queue) + self._running}")
        
        if not self._ttft_violation_imminent(current_time):
            print("No TTFT violation imminent")
            return
        
        candidates = self._collect_outsourcing_candidates()
        if not candidates:
            return

        items = [self._knapsack_item_for(r) for r in candidates]
        budget = self._local_prefill_budget_horizon()
        print(f"Budget {budget} tokens")

        keep_ids, outsource_ids = self._knapsack_select(items, budget)
        if outsource_ids:
            print(f"Outsourcing {(outsource_ids)} requests")
            self._apply_outsourcing(outsource_ids)

    # ---- Helpers ----
    def _ttft_violation_imminent(self, now: float) -> bool:
        head = self._waiting_queue.peek() if len(self._waiting_queue) else None
        if head is None:
            return False
        if head.prefill_slo_time is None:
            # If no per-request SLO is set, fall back to queue length heuristic.
            return len(self._waiting_queue) > self._max_micro_batch_size
        # Estimate TTFT for the head request under FCFS
        est_ttft = self._estimate_fcfs_ttft(head)
        print(f"Head req {head.id} est TTFT {est_ttft:.2f}s, deadline at {head.prefill_deadline_at:.2f}, now {now:.2f}")
        return (head.prefill_deadline_at - now) < est_ttft  # :contentReference[oaicite:11]{index=11}

    def _estimate_fcfs_ttft(self, req: Request) -> float:
        """
        Estimate Time-to-First-Token under FCFS assumption:
        = queueing delay (prefill of earlier requests) + own prefill time.
        """
        # Effective prefill throughput per step
        Sp = 1000
        # self._config.prefill_tokens_per_sec   # expose this in your config
        if Sp <= 0:
            return float("inf")

        # 1) Sum remaining prefill work of all waiting requests *ahead* of this one
        ahead_prefill = 0
        for r in self._waiting_queue.to_list():
            if r.id == req.id:
                break
            # account for sunk tokens (processed + cached)
            cached = self.get_cached_prefill_length(r)
            processed = r.num_processed_tokens
            prefill_done = max(processed, cached)
            rem = max(0, r.num_prefill_tokens - prefill_done)
            ahead_prefill += rem

        # 2) Own prefill work
        cached = self.get_cached_prefill_length(req)
        processed = req.num_processed_tokens
        prefill_done = max(processed, cached)
        rem_self = max(0, req.num_prefill_tokens - prefill_done)

        # 3) Convert to seconds
        est = (ahead_prefill + rem_self) / Sp
        return est

    def _iter_waiting_requests(self, limit: int | None = None):
        """
        Returns a stable snapshot list of waiting requests in FCFS order.
        Works for FCFSRequestQueue (has .to_list()) and degrades gracefully.
        """
        cands: list[Request] = []

        ls = self._waiting_queue.to_list() if hasattr(self._waiting_queue, "to_list") else list(self._waiting_queue)
        # Waiting requests (cheap to outsource) — cap to a small multiple of micro-batch size
        k = min(len(ls), 4 * self._max_micro_batch_size)
        for r in ls[:k]:
            cands.append(r)

        # Optionally add running requests that are still in prefill (avoid ejecting those in decode)
        for r in self._running:
            if not r.is_prefill_complete:
                cands.append(r)

        return cands

    def _collect_outsourcing_candidates(self) -> list[Request]:
        cands = []
        # Waiting requests (cheap to outsource)
        ls = self._waiting_queue.to_list() if hasattr(self._waiting_queue, "to_list") else list(self._waiting_queue)
        # Waiting requests (cheap to outsource) — cap to a small multiple of micro-batch size
        k = min(len(ls), 4 * self._max_micro_batch_size)
        for r in ls[:k]:
            cands.append(r)

        # Optionally add *running* requests that are still in prefill (avoid decoding-phase ejection)
        for r in self._running:
            if not r.is_prefill_complete:  # decoding-phase is bad UX to evict
                cands.append(r)
        return cands  # 

    def _knapsack_item_for(self, r: Request):
        cached = self.get_cached_prefill_length(r)         # prefix reuse (sunk) :contentReference[oaicite:13]{index=13}
        processed = r.num_processed_tokens                 # sunk compute      :contentReference[oaicite:14]{index=14}
        prefill_done = max(processed, cached)
        rem_prefill = max(0, r.num_prefill_tokens - prefill_done)
        decode_done = max(0, processed - r.num_prefill_tokens)
        rem_decode = max(0, r.num_decode_tokens - decode_done)
        alpha = 0.6
        # self._config.decode_weight_ratio  # expose Sp/Sd or equivalent
        weight = rem_prefill + alpha * rem_decode
        value = r.num_prefill_tokens  # or $-savings if you inject API prices
        print(f"  Knapsack item: req {r.id} weight {weight} value {value} (rem_prefill {rem_prefill} rem_decode {rem_decode})")
        return {"id": r.id, "weight": max(1, weight), "value": max(1, value)}

    def _local_prefill_budget_horizon(self) -> int:
        horizon = 2  # look 2 iterations ahead; tuneable
        return horizon * self._max_micro_batch_size * self._config.chunk_size  # :contentReference[oaicite:15]{index=15}

    def _knapsack_select(self, items, budget):
        # Greedy by value/weight: keep highest “bang-per-token” locally
        items = sorted(items, key=lambda x: x["value"]/x["weight"], reverse=True)
        keep, total = [], 0
        for it in items:
            if total + it["weight"] <= budget:
                keep.append(it["id"])
                total += it["weight"]
        keep_set = set(keep)
        outsource = [it["id"] for it in items if it["id"] not in keep_set]
        return keep_set, outsource

    def _apply_outsourcing(self, outsource_ids: list[str]) -> None:
        # 1) Remove from waiting queue
        if outsource_ids:
            # Build a filtered deque without outsourced IDs
            # Take a snapshot of all waiting requests
            snapshot = self._waiting_queue.to_list() if hasattr(self._waiting_queue, "to_list") else []
            # Filter out outsourced
            kept = [r for r in snapshot if r.id not in outsource_ids]
            self._waiting_queue.clear()
            for r in kept:
                self._waiting_queue.push(r)

        # 2) Preempt running prefill requests if selected
        new_running = []
        for r in self._running:
            if r.id in outsource_ids:
                # free KV, restart to normalize internal counters, then drop
                self._kv_cache_manager.free(r)
                r.restart()  # same behavior as existing preemption path 
                self._kv_cache_manager.free_block_hashes(r)
                self.scheduled_req_ids.discard(r.id)
                self._requests.pop(r.id, None)
                self._outsourced_req_ids.add(r.id)
            else:
                new_running.append(r)
        self._running = new_running
        
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
                if num_new_tokens == 0:
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
