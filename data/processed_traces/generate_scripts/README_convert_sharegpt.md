# Convert ShareGPT to Mooncake Trace Format

This script converts conversations from the ShareGPT Vicuna dataset into the Mooncake conversation trace format used by Vidur simulator.

## Requirements

The script requires the following packages (already installed in the Vidur venv):
- `datasets` (Hugging Face)
- `tiktoken` (for accurate token counting)
- `pandas`

## Usage

### Basic Usage

Convert 1000 conversations with default settings:

```bash
/raid/user_data/yiyanz/vidur/.venv/bin/python convert_sharegpt_to_mooncake.py \
    --output sharegpt_trace_1k.csv \
    --max_conversations 1000
```

### Full Dataset Conversion

Convert the entire dataset (this will take a while):

```bash
/raid/user_data/yiyanz/vidur/.venv/bin/python convert_sharegpt_to_mooncake.py \
    --output sharegpt_full_trace.csv \
    --streaming
```

### Custom Parameters

```bash
/raid/user_data/yiyanz/vidur/.venv/bin/python convert_sharegpt_to_mooncake.py \
    --output custom_trace.csv \
    --max_conversations 5000 \
    --block_size 16 \
    --time_interval_mean 2.0 \
    --time_interval_std 1.0 \
    --max_decode_tokens 512 \
    --min_prefill_tokens 10 \
    --seed 42 \
    --streaming
```

## Parameters

- `--output`: Output CSV file path (default: `sharegpt_mooncake_trace.csv`)
- `--max_conversations`: Maximum number of conversations to process (default: None = all)
- `--block_size`: Block size for KV cache simulation (default: 16)
- `--start_time`: Starting timestamp in seconds (default: 0.0)
- `--time_interval_mean`: Mean time between requests in seconds (default: 1.0)
- `--time_interval_std`: Standard deviation of time intervals (default: 0.5)
- `--seed`: Random seed for timestamp generation (default: 42)
- `--max_decode_tokens`: Cap maximum decode tokens per response (default: None)
- `--min_prefill_tokens`: Minimum prefill tokens to include a turn (default: 1)
- `--streaming`: Use streaming mode for large datasets (recommended for full dataset)

## Output Format

The output CSV has the following columns matching the Mooncake trace format:

- `arrived_at`: Request arrival timestamp (float, seconds)
- `num_prefill_tokens`: Number of tokens in the user prompt
- `num_decode_tokens`: Number of tokens in the assistant response
- `block_hash_ids`: JSON list of block hash IDs for prefix caching
- `block_size`: KV cache block size (typically 16)
- `session_id`: Conversation session identifier

## Examples

### Generate a trace with high QPS

```bash
/raid/user_data/yiyanz/vidur/.venv/bin/python convert_sharegpt_to_mooncake.py \
    --output high_qps_trace.csv \
    --max_conversations 10000 \
    --time_interval_mean 0.1 \
    --time_interval_std 0.05 \
    --streaming
```

### Generate a trace with longer responses capped

```bash
/raid/user_data/yiyanz/vidur/.venv/bin/python convert_sharegpt_to_mooncake.py \
    --output capped_trace.csv \
    --max_conversations 5000 \
    --max_decode_tokens 256 \
    --streaming
```

### Generate a small test trace

```bash
/raid/user_data/yiyanz/vidur/.venv/bin/python convert_sharegpt_to_mooncake.py \
    --output test_trace.csv \
    --max_conversations 10
```

## Notes

- The script uses `tiktoken` with the `cl100k_base` encoding (GPT-3.5/GPT-4 tokenizer) for accurate token counting
- Multi-turn conversations from the same session are preserved with the same `session_id`
- Block hash IDs are generated to simulate KV cache prefix sharing in multi-turn conversations
- Timestamps are generated with Gaussian noise to simulate realistic arrival patterns
- Use `--streaming` mode when processing large datasets to avoid memory issues

## Using the Generated Trace with Vidur

After generating a trace file, you can use it with Vidur simulator:

```bash
python -m vidur.main \
    --replica_config_model_name meta-llama/Meta-Llama-3-8B \
    --replica_config_device h100 \
    --request_generator_config_type trace \
    --trace_request_generator_config_trace_file ./data/processed_traces/sharegpt_trace_1k.csv \
    --replica_scheduler_config_type vllm_v1 \
    --cache_config_enable_prefix_caching
```
