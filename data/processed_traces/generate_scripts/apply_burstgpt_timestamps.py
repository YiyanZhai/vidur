#!/usr/bin/env python3
"""
Apply BurstGPT timestamps to ShareGPT trace file.

This script takes a ShareGPT trace file and a BurstGPT trace file,
and replaces the ShareGPT timestamps with BurstGPT timestamps sequentially.

Usage:
    python apply_burstgpt_timestamps.py \
        --sharegpt_trace ../sharegpt_full_trace_multi_turn.csv \
        --burstgpt_trace ../BurstGPT_1_trace.csv \
        --output ../sharegpt_burstgpt_timestamps.csv
"""

import argparse
import csv
import json
import pandas as pd
from pathlib import Path


def apply_burstgpt_timestamps(
    sharegpt_file: str,
    burstgpt_file: str,
    output_file: str,
    verbose: bool = False,
):
    """
    Apply BurstGPT timestamps to ShareGPT trace sequentially.
    
    Args:
        sharegpt_file: Path to ShareGPT trace CSV
        burstgpt_file: Path to BurstGPT trace CSV (with arrived_at column)
        output_file: Output CSV file path
        verbose: Print detailed progress information
    """
    print(f"[info] Loading ShareGPT trace: {sharegpt_file}")
    sharegpt_df = pd.read_csv(sharegpt_file)
    
    print(f"[info] Loading BurstGPT trace: {burstgpt_file}")
    burstgpt_df = pd.read_csv(burstgpt_file)
    # Filter out the top 1/5 of BurstGPT timestamps (keep bottom 4/5)
    num_to_keep = int(len(burstgpt_df) * 0.8)
    burstgpt_df = burstgpt_df.tail(num_to_keep).reset_index(drop=True)
    print(f"[info] Filtered BurstGPT trace: keeping {num_to_keep} timestamps (80%)")
    
    # Verify required columns
    if 'arrived_at' not in sharegpt_df.columns:
        raise ValueError("ShareGPT trace must have 'arrived_at' column")
    if 'arrived_at' not in burstgpt_df.columns:
        raise ValueError("BurstGPT trace must have 'arrived_at' column")
    
    num_sharegpt = len(sharegpt_df)
    num_burstgpt = len(burstgpt_df)
    
    print(f"[info] ShareGPT trace: {num_sharegpt} requests")
    print(f"[info] BurstGPT trace: {num_burstgpt} timestamps")
    
    if num_sharegpt > num_burstgpt:
        print(f"[warning] ShareGPT has more requests ({num_sharegpt}) than BurstGPT timestamps ({num_burstgpt})")
        print(f"[warning] Will only use first {num_burstgpt} ShareGPT requests")
        sharegpt_df = sharegpt_df.head(num_burstgpt)
    elif num_sharegpt < num_burstgpt:
        print(f"[info] Using first {num_sharegpt} BurstGPT timestamps")
    
    # Sort BurstGPT by timestamp to ensure sequential order
    burstgpt_df = burstgpt_df.sort_values('arrived_at').reset_index(drop=True)
    
    # Apply timestamps sequentially
    sharegpt_df['arrived_at'] = burstgpt_df['arrived_at'].head(len(sharegpt_df)).values
    
    # Sort by timestamp to maintain order
    sharegpt_df = sharegpt_df.sort_values('arrived_at').reset_index(drop=True)
    
    # Write output
    print(f"[info] Writing output to: {output_file}")
    sharegpt_df.to_csv(output_file, index=False)
    
    print(f"[done] Wrote {len(sharegpt_df)} records to {output_file}")
    
    # Print statistics
    min_time = sharegpt_df['arrived_at'].min()
    max_time = sharegpt_df['arrived_at'].max()
    time_span = max_time - min_time
    
    print("\n[stats] Timestamp Statistics:")
    print(f"  Time range: {min_time:.2f}s to {max_time:.2f}s")
    print(f"  Time span: {time_span:.2f}s ({time_span/3600:.2f} hours)")
    print(f"  Average rate: {len(sharegpt_df) / time_span:.2f} req/s")
    
    if verbose and 'num_prefill_tokens' in sharegpt_df.columns:
        avg_prefill = sharegpt_df['num_prefill_tokens'].mean()
        avg_decode = sharegpt_df['num_decode_tokens'].mean()
        print(f"\n[stats] Token Statistics:")
        print(f"  Average prefill tokens: {avg_prefill:.1f}")
        print(f"  Average decode tokens: {avg_decode:.1f}")
        
        # Show time distribution
        print(f"\n[stats] Timestamp Distribution:")
        print(f"  Min: {min_time:.2f}s")
        print(f"  25th percentile: {sharegpt_df['arrived_at'].quantile(0.25):.2f}s")
        print(f"  Median: {sharegpt_df['arrived_at'].median():.2f}s")
        print(f"  75th percentile: {sharegpt_df['arrived_at'].quantile(0.75):.2f}s")
        print(f"  Max: {max_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Apply BurstGPT timestamps to ShareGPT trace file"
    )
    parser.add_argument(
        "--sharegpt_trace",
        type=str,
        required=True,
        help="Path to ShareGPT trace CSV file",
    )
    parser.add_argument(
        "--burstgpt_trace",
        type=str,
        required=True,
        help="Path to BurstGPT trace CSV file (with arrived_at column)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed statistics",
    )
    
    args = parser.parse_args()
    
    apply_burstgpt_timestamps(
        args.sharegpt_trace,
        args.burstgpt_trace,
        args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
