# Outsourcing Configuration Guide

## Overview

The outsourcing feature in the VLLM V1 scheduler now has **7 configurable parameters** that control how requests are outsourced to external APIs. These parameters are defined in `VllmV1SchedulerConfig` and can be set via command-line arguments.

## Configuration Parameters

### 1. `prefill_tokens_per_sec`
- **Type**: `int`
- **Default**: `1000`
- **Description**: Estimated prefill throughput in tokens/sec used for TTFT violation detection
- **Command-line**: `--vllm_v1_scheduler_config_prefill_tokens_per_sec 1000`
- **Usage**: Higher values assume faster processing, which may reduce outsourcing frequency

### 2. `decode_weight_ratio`
- **Type**: `float`
- **Default**: `0.6`
- **Description**: Weight ratio for decode vs prefill tokens in knapsack optimization (decode_throughput / prefill_throughput)
- **Command-line**: `--vllm_v1_scheduler_config_decode_weight_ratio 0.6`
- **Usage**: Adjust based on your model's decode/prefill speed ratio. Lower values favor keeping decode-heavy requests local.

### 3. `budget_horizon_iterations`
- **Type**: `int`
- **Default**: `2`
- **Description**: Number of scheduling iterations to look ahead when calculating local processing budget
- **Command-line**: `--vllm_v1_scheduler_config_budget_horizon_iterations 2`
- **Usage**: Higher values = more conservative outsourcing (more budget reserved for local processing)

### 4. `candidate_queue_multiplier`
- **Type**: `int`
- **Default**: `4`
- **Description**: Maximum candidate queue size = multiplier × micro-batch size
- **Command-line**: `--vllm_v1_scheduler_config_candidate_queue_multiplier 4`
- **Usage**: Limits how many requests are considered for outsourcing. Higher = more candidates evaluated.

### 5. `knapsack_strategy`
- **Type**: `str`
- **Default**: `"dp_scaled"`
- **Description**: Algorithm for selecting which requests to keep local vs outsource
- **Command-line**: `--vllm_v1_scheduler_config_knapsack_strategy dp_scaled`
- **Options**:
  - `fractional`: Greedy algorithm, sorts by value/weight ratio (fastest)
  - `dp`: Exact 0/1 knapsack dynamic programming (optimal but slower)
  - `dp_scaled`: Scaled DP for large weights (recommended, balances speed & accuracy)
  - `random`: Random selection (baseline for testing)

### 6. `ttft_violation_mode`
- **Type**: `str`
- **Default**: `"all"`
- **Description**: How to detect TTFT (Time-To-First-Token) violations
- **Command-line**: `--vllm_v1_scheduler_config_ttft_violation_mode all`
- **Options**:
  - `all`: Check every waiting request for potential TTFT violations (conservative)
  - `head`: Only check head of FCFS queue (faster, less conservative)

### 7. `debug_outsourcing`
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable detailed debug logging for outsourcing decisions
- **Command-line**: `--vllm_v1_scheduler_config_debug_outsourcing`
- **Usage**: Add this flag to see detailed logs about outsourcing decisions in your simulation output

---

## How to Set Configuration

### Method 1: Command-Line Arguments (Recommended)

Add the parameters when running the simulator:

```bash
python -m vidur.main \
    --replica_scheduler_config_type vllm_v1 \
    --vllm_v1_scheduler_config_chunk_size 512 \
    --vllm_v1_scheduler_config_batch_size_cap 512 \
    --vllm_v1_scheduler_config_prefill_tokens_per_sec 1000 \
    --vllm_v1_scheduler_config_decode_weight_ratio 0.6 \
    --vllm_v1_scheduler_config_budget_horizon_iterations 2 \
    --vllm_v1_scheduler_config_candidate_queue_multiplier 4 \
    --vllm_v1_scheduler_config_knapsack_strategy dp_scaled \
    --vllm_v1_scheduler_config_ttft_violation_mode all \
    --vllm_v1_scheduler_config_debug_outsourcing \
    # ... other parameters
```

### Method 2: Modify Default Values in Code

Edit `vidur/config/config.py`:

```python
@dataclass
class VllmV1SchedulerConfig(BaseReplicaSchedulerConfig):
    chunk_size: int = field(
        default=512,
        metadata={"help": "Chunk size for chunked prefill."},
    )
    
    # Modify these defaults as needed
    prefill_tokens_per_sec: int = field(
        default=1500,  # Changed from 1000
        metadata={"help": "Estimated prefill throughput in tokens/sec."},
    )
    
    decode_weight_ratio: float = field(
        default=0.7,  # Changed from 0.6
        metadata={"help": "Weight ratio for decode vs prefill tokens."},
    )
    
    # ... etc
```

---

## Configuration Location

**File**: `vidur/config/config.py`

**Class**: `VllmV1SchedulerConfig` (starting at line 395)

**Parent Class**: `BaseReplicaSchedulerConfig`

**Access in Scheduler**: 
- In `vidur/scheduler/replica_scheduler/vllm_v1_replica_scheduler.py`, the config is available as `self._config`
- The config is set in the parent class `BaseReplicaScheduler.__init__()` at line 37:
  ```python
  self._config = replica_scheduler_config
  ```

---

## Example Configurations

### Conservative Outsourcing (Minimize API Costs)
```bash
--vllm_v1_scheduler_config_prefill_tokens_per_sec 1500 \
--vllm_v1_scheduler_config_budget_horizon_iterations 3 \
--vllm_v1_scheduler_config_knapsack_strategy dp_scaled \
--vllm_v1_scheduler_config_ttft_violation_mode head
```
- Higher throughput estimate → less likely to detect violations
- Larger horizon → more local budget
- Check only head → fewer violations detected

### Aggressive Outsourcing (Maximize Local Throughput)
```bash
--vllm_v1_scheduler_config_prefill_tokens_per_sec 800 \
--vllm_v1_scheduler_config_budget_horizon_iterations 1 \
--vllm_v1_scheduler_config_knapsack_strategy fractional \
--vllm_v1_scheduler_config_ttft_violation_mode all
```
- Lower throughput estimate → more violations detected
- Smaller horizon → less local budget
- Check all requests → maximum violations detected
- Fractional strategy → fastest algorithm

### Debugging Configuration
```bash
--vllm_v1_scheduler_config_debug_outsourcing \
--vllm_v1_scheduler_config_knapsack_strategy random
```
- Enable debug logs to see what's being outsourced
- Use random strategy as baseline for comparison

---

## Verification

After modifying the config, verify it's being used:

1. **Check syntax**:
   ```bash
   python -m py_compile vidur/config/config.py
   ```

2. **Run simulation with debug logging**:
   ```bash
   python -m vidur.main \
       --replica_scheduler_config_type vllm_v1 \
       --vllm_v1_scheduler_config_debug_outsourcing \
       # ... other params
   ```

3. **Check output files**:
   - `simulator_output/*/outsourced_requests.csv` - Should contain outsourced request details
   - `simulator_output/*/cluster_outsourcing_statistics.json` - Should show outsourcing stats
   - Logs should show lines like: `[Replica 0] TTFT violation detected at t=123.45`

---

## Parameter Tuning Recommendations

### For Cost Optimization
Focus on:
- `budget_horizon_iterations` (increase to keep more local)
- `ttft_violation_mode` (use 'head' for less outsourcing)
- `knapsack_strategy` (use 'dp_scaled' for optimal cost decisions)

### For Latency Optimization
Focus on:
- `prefill_tokens_per_sec` (decrease to detect more violations)
- `ttft_violation_mode` (use 'all' to catch all potential violations)
- `candidate_queue_multiplier` (increase to consider more candidates)

### For Throughput Optimization
Focus on:
- Balance between local processing and outsourcing
- Use realistic `prefill_tokens_per_sec` based on your hardware
- Adjust `decode_weight_ratio` based on your model's characteristics

---

## Technical Details

### Config Flow
1. Command-line args → parsed by argparse
2. Creates `VllmV1SchedulerConfig` instance with specified values
3. Passed to `VLLMV1ReplicaScheduler.__init__()` as `replica_scheduler_config`
4. Stored as `self._config` in base class
5. Accessed via `self._config.parameter_name` in `_init_outsourcing_config()`

### Default Resolution
The `getattr()` pattern in `_init_outsourcing_config()` provides fallback defaults:
```python
self._prefill_throughput = getattr(
    self._config, 'prefill_tokens_per_sec', 1000  # 1000 is fallback
)
```

This means:
- If `prefill_tokens_per_sec` is set in config → use that value
- If not set → use fallback value (1000)
- This provides backward compatibility with configs that don't have new parameters

---

## Related Files

1. **Config Definition**: `vidur/config/config.py` (line 395)
2. **Scheduler Implementation**: `vidur/scheduler/replica_scheduler/vllm_v1_replica_scheduler.py`
3. **Config Initialization**: `vidur/scheduler/replica_scheduler/base_replica_scheduler.py` (line 37)
4. **Main Entry Point**: `vidur/main.py`

---

## Troubleshooting

### "Unknown argument" error
- Make sure parameter name follows pattern: `--vllm_v1_scheduler_config_<parameter_name>`
- Check spelling matches exactly what's in `VllmV1SchedulerConfig`

### Parameters not taking effect
- Verify you're using `--replica_scheduler_config_type vllm_v1`
- Check the generated `config.json` in output directory to confirm values
- Enable `--vllm_v1_scheduler_config_debug_outsourcing` to see if config is being used

### Invalid strategy/mode error
- Check exact string values: `'fractional'`, `'dp'`, `'dp_scaled'`, `'random'` for knapsack
- Check exact string values: `'all'`, `'head'` for TTFT mode
- Strings are case-sensitive

---

## Summary

All outsourcing behavior is now controlled through `VllmV1SchedulerConfig` in `vidur/config/config.py`. You can:

1. **Set via command-line**: Add `--vllm_v1_scheduler_config_<param>` arguments
2. **Set via code**: Modify default values in `VllmV1SchedulerConfig` class
3. **Access in code**: Use `self._config.<param>` in the scheduler

The configuration system provides:
- ✅ Type checking (int, float, str, bool)
- ✅ Default values for all parameters
- ✅ Help text for documentation
- ✅ Automatic command-line argument generation
- ✅ Backward compatibility via `getattr()` fallbacks
