from vidur.config import ReplicaConfig
from vidur.entities import Request


class RequestFLOPCalculator:
    """
    Calculates FLOPs required for individual requests based on their remaining work.
    """

    def __init__(self, replica_config: ReplicaConfig):
        self._replica_config = replica_config
        self._model_config = replica_config.model_config

        # Get device TFLOPs capacity (convert to FLOPs per second)
        self._device_flops_per_second = replica_config.device_config.fp16_tflops * (10**12)  # TFLOPs to FLOPs

        # Calculate parameters per device (for MLP FLOPs)
        from vidur.utils.param_counter import ParamCounter
        param_counter = ParamCounter(replica_config)
        self._num_params_per_device = param_counter.get_num_parameters_per_device()

        # Attention parameters
        self._num_layers_per_device = (
            self._model_config.num_layers // replica_config.num_pipeline_stages
        )
        self._num_q_heads_per_device = (
            self._model_config.num_q_heads // replica_config.tensor_parallel_size
        )
        self._num_kv_heads_per_device = (
            self._model_config.num_kv_heads // replica_config.tensor_parallel_size
        )
        self._head_dimension = self._model_config.embedding_dim // self._model_config.num_q_heads

    def _get_mlp_flops(self, num_tokens: int) -> float:
        """Calculate FLOPs for MLP layers."""
        return 2 * num_tokens * self._num_params_per_device

    def _get_attention_flops(self, num_new_tokens: int, kv_cache_size: int) -> float:
        """
        Calculate FLOPs for attention layers.

        Args:
            num_new_tokens: Number of new tokens being processed
            kv_cache_size: Total KV cache size (new tokens + existing cache)
        """
        return (
            4  # for number of ops in attention (Q@K, softmax, softmax@V, output projection)
            * self._num_layers_per_device
            * self._num_q_heads_per_device
            * self._head_dimension
            * num_new_tokens  # q length (new tokens)
            * kv_cache_size   # kv length (total context)
        )

    def calculate_request_flops(self, request: Request, num_new_tokens: int) -> float:
        """
        Calculate total FLOPs needed to process num_new_tokens for this request.

        Args:
            request: The request object
            num_new_tokens: Number of new tokens to process

        Returns:
            Total FLOPs required
        """
        # MLP FLOPs (same for prefill and decode)
        mlp_flops = self._get_mlp_flops(num_new_tokens)

        # Attention FLOPs depend on whether we're in prefill or decode phase
        if request.num_processed_tokens < request.num_prefill_tokens:
            # Prefill phase: KV cache grows with new tokens
            kv_cache_size = num_new_tokens
        else:
            # Decode phase: KV cache includes all previous tokens + new token
            kv_cache_size = request.num_processed_tokens + num_new_tokens

        attention_flops = self._get_attention_flops(num_new_tokens, kv_cache_size)

        return mlp_flops + attention_flops

    def get_device_flops_budget_per_iteration(self, utilization_factor: float = 0.8) -> float:
        """
        Get the FLOP budget available per scheduling iteration.

        Args:
            utilization_factor: Target utilization (default 0.8 for 80%)

        Returns:
            FLOPs available per iteration
        """
        # Assume we know the scheduling interval - for now use a reasonable estimate
        # This could be made configurable or estimated from profiling data
        estimated_iteration_time_seconds = 0.1  # 100ms per iteration (tunable)

        return self._device_flops_per_second * estimated_iteration_time_seconds * utilization_factor
