# TTFT Estimation Tracking

## Overview

This feature tracks estimated Time-to-First-Token (TTFT) for each request in the waiting queue, enabling comparison with actual TTFT from simulation runs. This helps validate the accuracy of TTFT predictions used in outsourcing decisions.

## How It Works

### 1. **TTFTEstimateTracker** (New Module)
Located at: `vidur/scheduler/replica_scheduler/outsourcing/ttft_tracker.py`

Tracks estimated TTFT with rich metadata:
- `estimated_ttft`: Predicted time-to-first-token (seconds)
- `current_time`: Simulation time when estimate was made
- `deadline`: Prefill deadline (if SLO is set)
- `slack`: Time buffer before violation (deadline - current_time - estimated_ttft)
- `remaining_prefill_tokens`: Tokens left to process
- `queue_position`: Position in waiting queue
- `ahead_prefill_tokens`: Total tokens ahead in queue
- `is_violation`: Whether estimate exceeds deadline

### 2. **Integration with Violation Detector**
The `TTFTViolationDetector` now accepts an optional `ttft_tracker` parameter. When provided, it automatically records estimates for every request during violation checks.

### 3. **Automatic CSV Export**
At simulation end, the tracker saves estimates to:
```
<output_dir>/ttft_estimates_replica_<replica_id>.csv
```

### 4. **Cleanup on Outsourcing/Completion**
Estimates are cleared when requests are:
- Outsourced (no longer need tracking)
- Completed (already tracked)

## Usage

### Running a Simulation with TTFT Tracking

The feature is **automatically enabled** for VLLM V1 scheduler. Just run your simulation:

```bash
python -m vidur.main \
    --replica_scheduler_config_type vllm_v1 \
    --vllm_v1_scheduler_config_debug_outsourcing \
    ... (other args)
```

After the run, find estimates at:
```
simulator_output/<timestamp>/ttft_estimates_replica_0.csv
```

### Comparing Estimated vs Actual TTFT

**Step 1:** Run simulation WITHOUT outsourcing to get actual TTFT (ground truth):
```bash
python -m vidur.main --replica_scheduler_config_type vllm_v1 ... # without outsourcing
# Output: simulator_output/run_baseline/request_metrics.csv
```

**Step 2:** Run simulation WITH outsourcing to get estimated TTFT:
```bash
python -m vidur.main --replica_scheduler_config_type vllm_v1 ... # with outsourcing enabled
# Output: simulator_output/run_outsourcing/ttft_estimates_replica_0.csv
```

**Step 3:** Compare using the provided script:
```bash
python scripts/compare_ttft_estimates.py \
    --estimates simulator_output/run_outsourcing/ttft_estimates_replica_0.csv \
    --actual simulator_output/run_baseline/request_metrics.csv \
    --output ttft_comparison.png
```

## Output Files

### 1. `ttft_estimates_replica_<id>.csv`
Columns:
- `request_id`: Unique request identifier
- `estimated_ttft`: Estimated TTFT (seconds)
- `current_time`: When estimate was made
- `estimated_completion_time`: current_time + estimated_ttft
- `deadline`: Prefill deadline (or inf)
- `time_until_deadline`: deadline - current_time
- `slack`: Time buffer before violation
- `remaining_prefill_tokens`: Tokens left to prefill
- `queue_position`: Position in waiting queue (0-indexed)
- `ahead_prefill_tokens`: Total prefill tokens ahead
- `is_violation`: Boolean flag for deadline violation

### 2. `ttft_comparison.csv` (from comparison script)
All columns from estimates CSV plus:
- `actual_ttft`: Actual TTFT from baseline run
- `absolute_error`: actual_ttft - estimated_ttft
- `relative_error`: absolute_error / actual_ttft
- `percent_error`: relative_error * 100

### 3. `ttft_comparison.png` (from comparison script)
Four-panel plot:
1. **Scatter**: Estimated vs Actual TTFT (with perfect prediction line)
2. **Histogram**: Absolute error distribution
3. **Histogram**: Percent error distribution
4. **Scatter**: Error vs Queue Position (colored by estimated TTFT)

## Statistics Provided

The comparison script prints:
- Mean, median, std for estimated and actual TTFT
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Bias analysis (underestimation vs overestimation rates)

## Example Output

```
====================================================
TTFT COMPARISON STATISTICS
====================================================

Number of requests: 128

Estimated TTFT:
  Mean: 0.4523s
  Median: 0.3821s
  Std: 0.2156s

Actual TTFT:
  Mean: 0.4789s
  Median: 0.4012s
  Std: 0.2301s

Absolute Error (actual - estimated):
  Mean: 0.0266s
  Median: 0.0191s
  MAE: 0.0345s
  RMSE: 0.0512s

Percent Error:
  Mean: 5.56%
  Median: 4.76%
  MAPE: 7.21%

Estimation Bias:
  Underestimated (actual > estimated): 82 (64.1%)
  Overestimated (actual < estimated): 46 (35.9%)
====================================================
```

## Implementation Details

### Key Design Decisions

1. **Tracking Point**: Estimates are recorded during violation detection, not during scheduling. This captures the "decision point" when outsourcing is considered.

2. **Cleanup Strategy**: Estimates are cleared when requests are outsourced or completed to avoid memory bloat and prevent stale data.

3. **Optional Integration**: The tracker is optional - if not provided to `TTFTViolationDetector`, tracking is simply skipped (no errors).

4. **Per-Replica Files**: Each replica saves its own estimates file to support multi-replica simulations.

## Notes and Limitations

1. **Request ID Matching**: The comparison assumes request IDs are consistent between runs. Use the same random seed for reproducibility.

2. **SLO Requirement**: Full tracking data requires requests to have `prefill_slo_time` set. Without it, some fields (deadline, slack) will be `inf`.

3. **Queue Position**: Queue position is 0-indexed and reflects the order at the time of estimate, not at arrival.

4. **Timing**: Estimates are made at violation check time, which may be before the request actually starts execution.

5. **Multiple Estimates**: If a request is checked multiple times (across iterations), only the latest estimate is retained.

## Future Enhancements

Potential improvements:
- Track estimate history (multiple estimates per request)
- Add confidence intervals for estimates
- Support for non-FCFS scheduling policies
- Real-time comparison during simulation (if baseline data is available)
- Integration with W&B for automated tracking and visualization
