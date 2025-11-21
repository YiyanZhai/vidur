#!/bin/bash
# Script to generate ShareGPT trace with BurstGPT timestamps
# Usage: ./generate_trace.sh [max_conversations]

set -e  # Exit on error

# Configuration
MAX_CONVERSATIONS=${1:-10000}  # Default 10k conversations, or use first argument
OUTPUT_DIR="../"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

# File names
SHAREGPT_BASE="sharegpt_${MAX_CONVERSATIONS}conv_multi_turn.csv"
BURSTGPT_TRACE="../BurstGPT_1_trace.csv"
FINAL_OUTPUT="sharegpt_${MAX_CONVERSATIONS}conv_burstgpt_timestamps.csv"

echo "=================================================="
echo "ShareGPT to Mooncake Trace Generator"
echo "=================================================="
echo "Max conversations: ${MAX_CONVERSATIONS}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Step 1: Convert ShareGPT to Mooncake format with multi-turn
echo "[Step 1/2] Converting ShareGPT to Mooncake format..."
echo "  Command: convert_sharegpt_to_mooncake.py"
echo "  Output: ${OUTPUT_DIR}${SHAREGPT_BASE}"
echo ""

$PYTHON "${SCRIPT_DIR}/convert_sharegpt_to_mooncake.py" \
    --output "${OUTPUT_DIR}${SHAREGPT_BASE}" \
    --max_conversations ${MAX_CONVERSATIONS} \
    --block_size 16 \
    --start_time 0.0 \
    --time_interval_mean 1.0 \
    --time_interval_std 0.5 \
    --seed 42 \
    --streaming \
    --multi_turn

if [ $? -ne 0 ]; then
    echo "ERROR: ShareGPT conversion failed!"
    exit 1
fi

echo ""
echo "[Step 1/2] ✓ Complete"
echo ""

# Step 2: Apply BurstGPT timestamps
echo "[Step 2/2] Applying BurstGPT timestamps..."
echo "  Command: apply_burstgpt_timestamps.py"
echo "  BurstGPT trace: ${BURSTGPT_TRACE}"
echo "  Output: ${OUTPUT_DIR}${FINAL_OUTPUT}"
echo ""

$PYTHON "${SCRIPT_DIR}/apply_burstgpt_timestamps.py" \
    --sharegpt_trace "${OUTPUT_DIR}${SHAREGPT_BASE}" \
    --burstgpt_trace "${BURSTGPT_TRACE}" \
    --output "${OUTPUT_DIR}${FINAL_OUTPUT}" \
    --verbose

if [ $? -ne 0 ]; then
    echo "ERROR: BurstGPT timestamp application failed!"
    exit 1
fi

echo ""
echo "[Step 2/2] ✓ Complete"
echo ""

# Summary
echo "=================================================="
echo "Generation Complete!"
echo "=================================================="
echo "Final trace file: ${OUTPUT_DIR}${FINAL_OUTPUT}"
echo ""
echo "To use with Vidur simulator:"
echo "  --trace_request_generator_config_trace_file ./data/processed_traces/${FINAL_OUTPUT}"
echo ""

# Optional: Show trace statistics
echo "Trace Statistics:"
$PYTHON -c "
import pandas as pd
df = pd.read_csv('${OUTPUT_DIR}${FINAL_OUTPUT}')
print(f'  Total requests: {len(df):,}')
print(f'  Unique sessions: {df[\"session_id\"].nunique():,}')
print(f'  Avg prefill tokens: {df[\"num_prefill_tokens\"].mean():.1f}')
print(f'  Avg decode tokens: {df[\"num_decode_tokens\"].mean():.1f}')
print(f'  Time span: {df[\"arrived_at\"].max() - df[\"arrived_at\"].min():.0f} seconds ({(df[\"arrived_at\"].max() - df[\"arrived_at\"].min())/3600:.1f} hours)')
print(f'  Request rate: {len(df) / (df[\"arrived_at\"].max() - df[\"arrived_at\"].min()):.2f} req/s')
"

echo ""
echo "Done!"
