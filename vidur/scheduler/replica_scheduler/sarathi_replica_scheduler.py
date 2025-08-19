from math import ceil

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from vidur.logger import init_logger
logger = init_logger(__name__)

class SarathiReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # sarathi config
        self._num_running_batches = 0
        self._preempted_requests = []
        # For vLLM and its derivatives, we only need to set a loose max batch size
        # Memory requirements are handled explicitly by the scheduler
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages
        self._watermark_blocks = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks
        )

    def estimate_ttft_ms_if_enqueued_now(self, request: "Request") -> float:
        """
        Predict TTFT if `request` were added now (non-mutating).
        TTFT ≈ backlog_wait_ms + time_to_first_prefill_chunk_ms
        """
        # --- 0) Quick exit: if prefill is already done, TTFT is basically queueing ---
        remaining_prefill = max(0, request.num_prefill_tokens - request.num_processed_tokens)
        if remaining_prefill == 0:
            return self._estimate_backlog_ms(avg_first_chunk_ms=50.0) + 1.0

        # --- 1) Memory feasibility now: can we admit a *new* request immediately? ---
        # Recompute the same condition as _can_allocate_request() for a *new* request
        req_blocks = ceil(request.num_prefill_tokens / self._config.block_size)
        free_blocks_now = self._config.num_blocks - self._num_allocated_blocks
        can_admit_now = (free_blocks_now - req_blocks) >= self._watermark_blocks

        # If not, estimate time to free the deficit
        deficit_blocks = max(0, (self._watermark_blocks + req_blocks) - free_blocks_now)
        wait_for_blocks_ms = 0.0
        if not can_admit_now:
            # Conservative: each freed block ~ one "chunk window" of time.
            # Use a running avg if you keep one; else a safe fallback.
            avg_chunk_ms = getattr(self, "_moving_avg_first_chunk_ms", None) or 150.0
            wait_for_blocks_ms = deficit_blocks * avg_chunk_ms

        # --- 2) Additional queueing: number of requests ahead of us that will insert a chunk before we do ---
        # Preempted requests likely to resume first, then new arrivals queued.
        ahead = 0
        if hasattr(self, "_preempted_requests"):
            # Only those that can likely emit a chunk soon (prefill complete first)
            # Keep it conservative: count all preempted as ahead.
            ahead += len(self._preempted_requests)
        if hasattr(self, "_request_queue"):
            ahead += len(self._request_queue)

        # Time slice per "ahead" request until we get our first slice.
        # Again, use running average if available; fallback conservative.
        avg_first_chunk_ms = getattr(self, "_moving_avg_first_chunk_ms", None) or 150.0
        wait_for_turn_ms = ahead * avg_first_chunk_ms

        # --- 3) Time to produce our first chunk (prefill) once we start ---
        # Sarathi emits early after a watermark of blocks; also limited by chunk_size.
        watermark_tokens = max(1, self._watermark_blocks * self._config.block_size)
        first_chunk_tokens = min(remaining_prefill, self._config.chunk_size, watermark_tokens)

        # Build a synthetic single-request batch for predictor
        tmp_batch = Batch(self._replica_id, [request], [first_chunk_tokens])

        stage0_sched = self._replica_stage_schedulers[0]
        exec_time = stage0_sched._execution_time_predictor.get_execution_time(
            tmp_batch, pipeline_stage=0
        )

        # Prefer consolidated model_time_ms if present; otherwise sum prefill-path pieces.
        if hasattr(exec_time, "model_time_ms") and exec_time.model_time_ms is not None:
            first_chunk_ms = float(exec_time.model_time_ms)
        else:
            per_layer_ms = (
                exec_time.attention_prefill_execution_time
                + exec_time.attention_pre_proj_time
                + exec_time.attention_post_proj_time
                + exec_time.mlp_layer_up_proj_execution_time
                + exec_time.mlp_layer_act_execution_time
                + exec_time.mlp_layer_down_proj_execution_time
                + exec_time.attention_kv_cache_save_execution_time
                + exec_time.attention_rope_execution_time
                + exec_time.add_time * 2.0
                + exec_time.attn_norm_time
                + exec_time.mlp_norm_time
                + exec_time.attention_all_reduce_time
                + exec_time.mlp_all_reduce_time
            )
            layers = getattr(exec_time, "num_layers", 1)
            first_chunk_ms = float(per_layer_ms) * float(layers)
            first_chunk_ms += float(exec_time.pipeline_parallel_communication_time)
            first_chunk_ms += float(exec_time.tensor_parallel_communication_time)
            first_chunk_ms += float(exec_time.schedule_time) + float(exec_time.prepare_inputs_e2e_time)

        backlog_ms = self._estimate_backlog_ms(avg_first_chunk_ms=avg_first_chunk_ms)
        res = backlog_ms + wait_for_blocks_ms + wait_for_turn_ms + first_chunk_ms
        # logger.info(
        #     f"TTFT est for req {request.id} (replica {self._replica_id}): "
        #     f"backlog {backlog_ms:.1f} + wait_blocks {wait_for_blocks_ms:.1f} + "
        #     f"wait_turn {wait_for_turn_ms:.1f} + first_chunk {first_chunk_ms:.1f} = {res:.1f} ms"
        # )
        return res


    def _estimate_backlog_ms(self, avg_first_chunk_ms: float = 150.0) -> float:
        """
        Estimate time until the replica can *start* another first-chunk window,
        based on currently running batches / stage-0 occupancy. Non-mutating.
        """
        # If stage schedulers are exposed, use stage-0 queue + running time
        stage_scheds = getattr(self, "_replica_stage_schedulers", None)
        if stage_scheds:
            try:
                stage0 = stage_scheds[0]
                total_ms = 0.0

                running = getattr(stage0, "current", None) or getattr(stage0, "_current", None)
                if running is not None:
                    rem = getattr(running, "remaining_ms", None)
                    if rem is None:
                        rem = getattr(running, "execution_time", 0.0)
                    total_ms += float(rem)

                q = getattr(stage0, "queue", None) or getattr(stage0, "_queue", None)
                if q is not None:
                    for bs in list(q):
                        total_ms += float(getattr(bs, "execution_time", 0.0))
                return total_ms
            except Exception:
                pass  # fall back

        # Fallback: coarse but safe—each running batch likely consumes one chunk window.
        running = getattr(self, "_num_running_batches", 0) or 0
        return float(running * avg_first_chunk_ms)
    
    def _can_allocate_request(self, request: Request) -> bool:
        if request.id not in self._allocation_map:
            # new request
            num_required_blocks = ceil(
                request.num_prefill_tokens / self._config.block_size
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
                request.num_prefill_tokens / self._config.block_size
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

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        for request in batch.requests:
            if request.completed:
                self.free(request.id)
            else:
                self._preempted_requests.append(request)

    def _get_request_next_num_tokens(
        self, request: Request, batch_contains_prefill: bool, num_batch_tokens: int
    ) -> int:
        assert not request.completed

        if request.is_prefill_complete:
            return 1

        next_num_tokens = min(
            request.num_prefill_tokens - request.num_processed_tokens,
            self._config.chunk_size - num_batch_tokens,
        )

        next_num_tokens = max(0, next_num_tokens)

        return next_num_tokens

    def _get_next_batch(self) -> Batch:
        requests = []
        num_tokens = []
        skipped_requests = []
        running_prefills = []
        contains_prefill = False
        num_batch_tokens = 0

        # preempted requests could contain multiple requests which have
        # partial prefills completed, so we need to be careful
        while self._preempted_requests:
            if len(requests) == self._max_micro_batch_size:
                break

            request = self._preempted_requests.pop(0)

            if not request.is_prefill_complete:
                running_prefills.append(request)
                continue

            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                skipped_requests.append(request)
                continue

            while not self._can_allocate_request(request):
                if self._preempted_requests:
                    victim_request = self._preempted_requests.pop(-1)
                    victim_request.restart()
                    self.free(victim_request.id)
                    self._request_queue = [victim_request] + self._request_queue
                else:
                    request.restart()
                    self.free(request.id)
                    self._request_queue = [request] + self._request_queue
                    break
            else:
                self._allocate_request(request)
                assert request.is_prefill_complete
                num_batch_tokens += next_num_tokens
                requests.append(request)
                num_tokens.append(next_num_tokens)

        for request in running_prefills:
            assert not request.is_prefill_complete

            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                skipped_requests.append(request)
                continue

            contains_prefill = True
            num_batch_tokens += next_num_tokens
            requests.append(request)
            num_tokens.append(next_num_tokens)

        # re-add the skipped requests, but make sure that we add them to the
        # front of the queue so that they are scheduled first and we maintain FIFO ordering
        self._preempted_requests = skipped_requests + self._preempted_requests
        self._preempted_requests = sorted(
            self._preempted_requests, key=lambda req: req.arrived_at
        )
        skipped_requests = []

        while self._request_queue:
            if len(self._allocation_map) == self._config.batch_size_cap:
                break

            if len(requests) == self._max_micro_batch_size:
                break

            if not self._can_allocate_request(self._request_queue[0]):
                break

            next_num_tokens = self._get_request_next_num_tokens(
                self._request_queue[0], contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                break

            request = self._request_queue.pop(0)

            self._allocate_request(request)

            # all new requests will have a prefill
            contains_prefill = True
            num_batch_tokens += next_num_tokens
            requests.append(request)
            num_tokens.append(next_num_tokens)

        if not requests:
            return

        return Batch(self._replica_id, requests, num_tokens)
