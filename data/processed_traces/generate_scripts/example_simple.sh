#!/bin/bash
# Simple example: Generate trace with 1000 conversations

cd /raid/user_data/yiyanz/vidur/data/processed_traces/generate_scripts

# Step 1: Convert ShareGPT to Mooncake format (multi-turn mode)
python convert_sharegpt_to_mooncake.py \
    --output ../sharegpt_1k_trace_multi_turn.csv \
    --max_conversations 1000 \
    --streaming \
    --multi_turn

# Step 2: Apply BurstGPT timestamps
python apply_burstgpt_timestamps.py \
    --sharegpt_trace ../sharegpt_1k_trace_multi_turn.csv \
    --burstgpt_trace ../BurstGPT_1_trace.csv \
    --output ../sharegpt_1k_burstgpt_timestamps.csv \
    --verbose

echo "Done! Generated: ../sharegpt_1k_burstgpt_timestamps.csv"
