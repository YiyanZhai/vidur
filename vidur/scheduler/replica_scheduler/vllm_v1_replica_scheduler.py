import math

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
from vidur.utils.request_flop_calculator import RequestFLOPCalculator


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

        # Create FLOP calculator for request cost estimation
        self._flop_calculator = RequestFLOPCalculator(self._replica_config)

        # req_id -> Request
        self._requests: Dict[str, Request] = {}
        # self._waiting_queue has been initialized in the parent class
        self._running: List[Request] = []
        # The requests that have been scheduled and are being executed
        # by the executor.
        self.scheduled_req_ids: set[str] = set()
        self._outsourced_req_ids: set[(str, bool)] = set()
        self._knapsack_select = self._knapsack_select_fractional
        # self._knapsack_select = self._knapsack_select_dp
        # self._knapsack_select = self._knapsack_select_dp_scaled
        # self._knapsack_select = self._sanity_check_randomly_select
        # self._exist_ttft_violation_ = self._ttft_violation_imminent
        self._exist_ttft_violation_ = self._ttft_violations_any
        
    # ---- Outsourcing hook ----
    def _maybe_outsource_before_schedule(self, current_time: float) -> None:
        if not len(self._waiting_queue) and not self._running:
            return

        while self._exist_ttft_violation_(current_time):
            candidates = self._collect_outsourcing_candidates()
            if not candidates:
                return

            items = [self._knapsack_item_for(r) for r in candidates]
            min_weight = min(item["weight"] for item in items) if items else 0
            budget = max(0, sum(item["weight"] for item in items) - min_weight)
            print(f"Budget {budget} tokens")

            keep_ids, outsource_ids = self._knapsack_select(items, budget)
            if outsource_ids:
                print(f"Outsourcing: {(outsource_ids)}")
                self._apply_outsourcing(outsource_ids)
                
        print(f"Remaining requests: {[r.id for r in self._waiting_queue.to_list()]}")

        # Below is the old code, where we do a single check of TTFT violations, and do normal knapsack
        # if not self._exist_ttft_violation_(current_time):
        #     # print("No TTFT violation imminent")
        #     return
        
        # candidates = self._collect_outsourcing_candidates()
        # if not candidates:
        #     return

        # # print(f"Considering outsourcing: {[c.id for c in candidates]}")

        # items = [self._knapsack_item_for(r) for r in candidates]
        # # budget = max(0, sum(item["weight"] for item in items) - 1)
        # budget = self._local_prefill_budget_horizon()
        # print(f"Budget {budget} tokens")

        # keep_ids, outsource_ids = self._knapsack_select(items, budget)
        # if outsource_ids:
        #     print(f"Outsourcing: {(outsource_ids)}")
        #     self._apply_outsourcing(outsource_ids)

    # ---- Helpers ----
    def _ttft_violations_any(self, now: float):
        """
        Check EVERY waiting request for imminent TTFT violation under FCFS.
        Returns: (any_violation: bool, at_risk_ids: set[str])
        """
        # Snapshot waiting queue (never iterate the queue object directly)
        waiting = self._iter_waiting_requests()
        if not waiting:
            return False

        # Prefill throughput (tokens/sec) — expose this in your scheduler config
        # Sp = getattr(self._config, "prefill_tokens_per_sec", None)
        # if not Sp or Sp <= 0:
        #     # Can't estimate ⇒ be conservative: no outsourcing trigger here
        #     return False, set()
        Sp = 1000

        # Precompute remaining prefill for each request (sunk work + prefix cache)
        rem_prefill = []
        for r in waiting:
            cached = self.get_cached_prefill_length(r)
            processed = r.num_processed_tokens
            prefill_done = max(processed, cached)
            rem = max(0, r.num_prefill_tokens - prefill_done)
            rem_prefill.append(rem)

        # Prefix sum: work ahead of each request in FCFS order
        ahead = [0] * len(waiting)
        acc = 0
        for i in range(len(waiting)):
            ahead[i] = acc
            acc += rem_prefill[i]
            
        # print(f"TTFT check at {now:.2f}s: reqs {[r.id for r in waiting]}, rem_prefill {rem_prefill}, ahead {ahead}")

        # Evaluate every request with an SLO; collect those at risk
        at_risk = set()
        saw_any_slo = False
        for i, r in enumerate(waiting):
            # If no explicit SLO on this request, derive a reasonable default
            # proportional to the prefill length. Formula used:
            #   default_slo = base_latency + slack_factor * (num_prefill_tokens / Sp)
            # where Sp is the estimated prefill throughput (tokens/sec) used above.
            slo_time = getattr(r, "prefill_slo_time", None)
            if slo_time is None:
                # Tunable defaults; can be exposed in scheduler config later
                base_latency = getattr(self._config, "prefill_slo_base_seconds", 0.05)
                slack_factor = getattr(self._config, "prefill_slo_slack_factor", 1.5)
                slo_time = base_latency + (r.num_prefill_tokens / max(1, Sp)) * slack_factor
                # Derive a deadline timestamp if not present. Prefer request.queued_at if available.
                queued_at = getattr(r, "queued_at", now)
                # Attach a derived deadline to the request so later logic that expects
                # `prefill_deadline_at` can work. This mutates the request but mirrors
                # behaviour when an explicit SLO is set elsewhere.
                if getattr(r, "prefill_deadline_at", None) is None:
                    try:
                        r.prefill_deadline_at = queued_at + slo_time
                    except Exception:
                        # If the Request object doesn't allow setting attributes, just
                        # fall back to computing time_left locally below.
                        pass
            else:
                # Explicit SLO given; ensure deadline timestamp exists (derive if missing).
                if getattr(r, "prefill_deadline_at", None) is None:
                    queued_at = getattr(r, "queued_at", now)
                    try:
                        r.prefill_deadline_at = queued_at + slo_time
                    except Exception:
                        pass

            saw_any_slo = True

            est_ttft = (ahead[i] + rem_prefill[i]) / Sp
            # deadline = queued_at + prefill_slo_time (we either set or derived prefill_deadline_at)
            time_left = getattr(r, "prefill_deadline_at", None) - now if getattr(r, "prefill_deadline_at", None) is not None else slo_time
            if est_ttft > time_left:
                # print(f"  Req {r.id} at risk: est TTFT {est_ttft:.2f}s > time left {time_left:.2f}s (deadline at {getattr(r, 'prefill_deadline_at', now + slo_time):.2f})")
                at_risk.add(r.id)

        # If none had an explicit SLO, fall back to a simple pressure heuristic
        if not saw_any_slo:
            # Example heuristic: too many waiting vs micro-batch capacity ⇒ treat as 'at risk'
            # (Avoid len(self._waiting_queue); work off the snapshot length)
            if len(waiting) > getattr(self, "_max_micro_batch_size", 1):
                # mark the first few as at risk to nudge outsourcing
                at_risk.update(r.id for r in waiting[: self._max_micro_batch_size])

        res = len(at_risk) > 0
        print(res)
        return res

    def _ttft_violation_imminent(self, now: float) -> bool:
        head = self._waiting_queue.peek() if len(self._waiting_queue) else None
        if head is None:
            return False
        if head.prefill_slo_time is None:
            # If no per-request SLO is set, fall back to queue length heuristic.
            return len(self._waiting_queue) > self._max_micro_batch_size
        # Estimate TTFT for the head request under FCFS
        est_ttft = self._estimate_fcfs_ttft(head)
        # print(f"Head req {head.id} est TTFT {est_ttft:.2f}s, deadline at {head.prefill_deadline_at:.2f}, now {now:.2f}")
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
        # # Waiting requests (cheap to outsource) — cap to a small multiple of micro-batch size
        # k = min(len(ls), 4 * self._max_micro_batch_size)
        k = len(ls)
        for r in ls[:k]:
            cands.append(r)

        # # Optionally add *running* requests that are still in prefill (avoid decoding-phase ejection)
        # for r in self._running:
        #     if not r.is_prefill_complete:  # decoding-phase is bad UX to evict
        #         cands.append(r)
        return cands  # 

    def _knapsack_item_for(self, r: Request):
        cached = self.get_cached_prefill_length(r)         # prefix reuse (sunk) :contentReference[oaicite:13]{index=13}
        processed = r.num_processed_tokens                 # sunk compute      :contentReference[oaicite:14]{index=14}
        prefill_done = max(processed, cached)
        rem_prefill = max(0, r.num_prefill_tokens - prefill_done)
        decode_done = max(0, processed - r.num_prefill_tokens)
        rem_decode = max(0, r.num_decode_tokens - decode_done)

        # Calculate FLOPs needed for remaining work
        weight = 0.0
        if rem_prefill > 0:
            # FLOPs for remaining prefill tokens
            weight += self._flop_calculator.calculate_request_flops(r, rem_prefill)
        if rem_decode > 0:
            # FLOPs for remaining decode tokens (process one at a time for decode)
            # For decode, we need to process tokens sequentially
            remaining_decode_tokens = list(range(rem_decode))
            for i in range(len(remaining_decode_tokens)):
                # Each decode step processes 1 token but KV cache grows
                weight += self._flop_calculator.calculate_request_flops(r, 1)

        # Use OpenAI API pricing for value calculation (revenue potential)
        # gpt-5: $1.25 per 1M input tokens, $10.00 per 1M output tokens
        input_price_per_token = 1.25 / 1_000_000
        output_price_per_token = 10.00 / 1_000_000
        value = r.num_prefill_tokens * input_price_per_token + r.num_decode_tokens * output_price_per_token

        # print(f"  Knapsack item: req {r.id} weight {weight:.2e} FLOPs value {value:.6f} (rem_prefill {rem_prefill} rem_decode {rem_decode})")
        return {"id": r.id, "weight": max(1, weight), "value": max(1e-10, value)}

    def _local_prefill_budget_horizon(self) -> float:
        """
        Calculate FLOP budget available for local processing before outsourcing.
        Uses GPU TFLOPs capacity * utilization factor.
        """
        utilization_factor = 0.8  # 80% utilization target
        return self._flop_calculator.get_device_flops_budget_per_iteration(utilization_factor)

    def _knapsack_select_fractional(self, items, budget):
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
    
    def _knapsack_select_dp(self, items, budget):
        """
        0/1 knapsack (dynamic programming).
        items: list[{"id": <hashable>, "weight": int>=1, "value": int>=0}]
        budget: int >= 0
        Returns: (keep_set, outsource_ids)
        """
        if budget <= 0 or not items:
            return set(), [it["id"] for it in items]

        n = len(items)
        # dp[b] = max value achievable with capacity b using items[0..i] (rolling over i)
        dp = [0] * (budget + 1)
        # choice[i][b] = True if item i is taken when achieving dp at capacity b
        choice = [[False] * (budget + 1) for _ in range(n)]

        # Fill DP
        for i, it in enumerate(items):
            w = int(it["weight"])
            v = int(it["value"])
            if w <= 0:
                # guard against bad inputs; treat as minimal weight
                w = 1
            if w > budget:
                # can't ever fit; skip updates but keep False in choice
                continue
            # iterate backward to avoid reusing item i more than once
            for b in range(budget, w, -1):
                if dp[b - w] + v > dp[b]:
                    dp[b] = dp[b - w] + v
                    choice[i][b] = True
            # handle b == w explicitly (range(...) excludes the endpoint)
            if dp[w] < v:
                dp[w] = v
                choice[i][w] = True

        # Reconstruct chosen set (take the best capacity)
        b = max(range(budget + 1), key=lambda x: dp[x])
        keep_ids = []
        for i in range(n - 1, -1, -1):
            if choice[i][b]:
                keep_ids.append(items[i]["id"])
                b -= int(items[i]["weight"]) if items[i]["weight"] > 0 else 1

        keep_set = set(keep_ids)
        outsource_ids = [it["id"] for it in items if it["id"] not in keep_set]
        return keep_set, outsource_ids

    def _knapsack_select_dp_scaled(self, items, budget, target_scaled_budget=5000, fallback_threshold=5_000_000):
        """
        Scaled 0/1 knapsack DP to handle very large weights/budgets.
        items: list[{"id": <hashable>, "weight": number>=1, "value": number>=0}]
        budget: number >= 0  (original units)
        target_scaled_budget: aim to shrink budget to about this size
        fallback_threshold: if n * scaled_budget exceeds this, fallback to greedy

        Returns: (keep_set, outsource_ids)
        """
        n = len(items)
        if budget <= 0 or n == 0:
            return set(), [it["id"] for it in items]

        # --- 1) Choose scale factor so scaled_budget ~ target_scaled_budget
        # scale >= 1; larger scale -> smaller scaled_budget
        # Handle FLOP values by first scaling them to reasonable integer range
        max_weight = max(it["weight"] for it in items) if items else 1
        flop_scale = max(1, max_weight / 1e6)  # Scale FLOPs to ~1M units
        scaled_budget_float = budget / flop_scale
        scale = max(1, math.ceil(scaled_budget_float / max(1, target_scaled_budget)))
        scaled_budget = max(1, int(scaled_budget_float // scale))

        # Helper: ceil_div for weights so we don't under-estimate capacity usage
        def ceil_div(a, b):  # b > 0
            return (a + b - 1) // b

        # --- 2) Build scaled items
        scaled_items = []
        for it in items:
            # Scale FLOP weights to integers
            w = max(1, int(float(it["weight"]) / flop_scale))
            v = max(0, int(float(it["value"]) * 1e6))  # Scale value for precision
            sw = max(1, ceil_div(w, scale))  # ceil divide to avoid underpacking
            scaled_items.append({"id": it["id"], "weight": sw, "value": v, "orig_weight": it["weight"]})

        # --- 3) If DP would be too large, fallback to greedy by value/weight
        if n * scaled_budget > fallback_threshold:
            # Greedy approximation as a safety valve
            ranked = sorted(scaled_items, key=lambda x: x["value"] / max(1, x["weight"]), reverse=True)
            keep, total_sw = [], 0
            for it in ranked:
                if total_sw + it["weight"] <= scaled_budget:
                    keep.append(it)
                    total_sw += it["weight"]
            keep_ids = set(it["id"] for it in keep)
            # Repair for original budget (rarely needed)
            keep_ids = self._repair_for_original_budget(keep_ids, items, budget)
            outsource_ids = [it["id"] for it in items if it["id"] not in keep_ids]
            return keep_ids, outsource_ids

        # --- 4) Standard 0/1 knapsack DP on scaled instance
        dp = [0] * (scaled_budget + 1)
        choice = [[False] * (scaled_budget + 1) for _ in range(n)]

        for i, it in enumerate(scaled_items):
            w = it["weight"]
            v = it["value"]
            if w > scaled_budget:
                continue
            # Backwards iteration for 0/1 knapsack
            for b in range(scaled_budget, w, -1):
                if dp[b - w] + v > dp[b]:
                    dp[b] = dp[b - w] + v
                    choice[i][b] = True
            # handle exactly b == w (range excludes endpoint)
            if dp[w] < v:
                dp[w] = v
                choice[i][w] = True

        # Reconstruct selection at best capacity
        b = max(range(scaled_budget + 1), key=lambda x: dp[x])
        keep_idx = []
        for i in range(n - 1, -1, -1):
            if b < 0:
                break
            if b <= scaled_budget and choice[i][b]:
                keep_idx.append(i)
                b -= scaled_items[i]["weight"]
                if b <= 0:
                    break

        keep_ids = set(scaled_items[i]["id"] for i in keep_idx)

        # --- 5) Repair step in ORIGINAL units (ensure feasibility w.r.t true budget)
        keep_ids = self._repair_for_original_budget(keep_ids, items, budget)

        outsource_ids = [it["id"] for it in items if it["id"] not in keep_ids]
        return keep_ids, outsource_ids
    
    def _sanity_check_randomly_select(self, items, budget):
        """
        Sanity check: randomly select items until budget is met.
        Should be similar to knapsack result on average.
        """
        import random

        ids = [it["id"] for it in items]
        random.shuffle(ids)
        total_w = 0
        selected = set()
        for id in ids:
            it = next(it for it in items if it["id"] == id)
            w = max(1, math.ceil(float(it["weight"])))
            if total_w + w <= budget:
                selected.add(id)
                total_w += w
            if total_w >= budget:
                break
        # print(f"Random selection kept {len(selected)}/{len(items)} items")
        outsource_ids = [it["id"] for it in items if it["id"] not in selected]
        return selected, outsource_ids

    def _repair_for_original_budget(self, keep_ids, items, budget):
        """
        If the scaled solution slightly exceeds the true (unscaled) budget due to ceil rounding,
        drop items with the worst value/weight ratio until the original budget is satisfied.
        """
        # Build list of kept items with original weights/values
        kept = [it for it in items if it["id"] in keep_ids]
        # Use ceil for weights to match the scaling logic (weights are already FLOPs)
        total_w = sum(max(1, math.ceil(float(it["weight"]))) for it in kept)
        
        if total_w <= budget:
            return keep_ids

        # Sort kept items by "weakness": lowest value/weight first
        # Drop until within true budget
        kept_sorted = sorted(
            kept,
            key=lambda it: ( (float(it["value"]) / max(1, math.ceil(float(it["weight"])))) if float(it["weight"]) > 0 else float('inf') )
        )
        
        keep_ids = set(keep_ids)
        for it in kept_sorted:
            if total_w <= budget:
                break
            item_weight = max(1, math.ceil(float(it["weight"])))
            keep_ids.discard(it["id"])
            total_w -= item_weight

        return keep_ids

    def _apply_outsourcing(self, outsource_ids: list[str]) -> None:
        # 1) Remove from waiting queue
        if outsource_ids:
            self._outsourced_req_ids.update((id, False) for id in outsource_ids)
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
                self._outsourced_req_ids.add((r.id, True))
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

        print(f"Replica {self._replica_id} scheduling: running {[r.id for r in self._running]}, waiting {[r.id for r in self._waiting_queue.to_list()] if self._waiting_queue else []}, scheduled {[(request.id, num_scheduled_tokens[request.id]) for request in scheduled_reqs]}, outsourced {len(self._outsourced_req_ids)}, mem {self.memory_usage_percent:.1f}%")

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
