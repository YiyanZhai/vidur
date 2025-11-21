#!/usr/bin/env python3
"""
Convert BurstGPT CSV format to Vidur trace format with prefix caching support.

BurstGPT Format:
    Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type

Vidur Trace Format:
    arrived_at,num_prefill_tokens,num_decode_tokens,block_hash_ids,block_size,session_id

Usage:
    python scripts/convert_burstgpt_to_trace.py \
        --input data/processed_traces/BurstGPT_1.csv \
        --output data/processed_traces/BurstGPT_1_trace.csv \
        --block-size 16 \
        --generate-block-hashes
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import List


def generate_block_hash_ids(num_prefill_tokens: int, num_decode_tokens: int, block_size: int) -> List[int]:
    """
    Generate block hash IDs for the entire request (prefill + decode tokens).
    
    For simplicity, we'll generate sequential block IDs. In a real scenario with prefix caching,
    these would be content-based hashes that allow sharing of common prefixes.
    
    Args:
        num_prefill_tokens: Number of prefill tokens
        num_decode_tokens: Number of decode tokens
        block_size: Size of each block (must be 16 for Vidur)
        
    Returns:
        List of block hash IDs
    """
    total_tokens = num_prefill_tokens + num_decode_tokens
    num_blocks = (total_tokens + block_size - 1) // block_size  # Ceiling division
    
    # Generate sequential block IDs (in a real scenario, these would be content hashes)
    # We'll use a simple counter that gets incremented globally
    return list(range(num_blocks))


def convert_burstgpt_to_trace(
    input_file: str,
    output_file: str,
    block_size: int = 16,
    generate_block_hashes: bool = True,
    session_id_column: str = None,
) -> None:
    """
    Convert BurstGPT CSV format to Vidur trace format.
    
    Args:
        input_file: Path to input BurstGPT CSV file
        output_file: Path to output trace CSV file
        block_size: Block size for prefix caching (default: 16)
        generate_block_hashes: Whether to generate block_hash_ids (default: True)
        session_id_column: Optional column name in BurstGPT to use as session_id
    """
    print(f"Reading BurstGPT file: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Filter to only keep ChatGPT model
    print(f"Filtering for ChatGPT model only...")
    original_count = len(df)
    df = df[df['Model'] == 'GPT-4']
    print(f"Filtered from {original_count:,} to {len(df):,} rows")
    
    # Normalize arrival times to start from 0 and convert to seconds
    df['arrived_at'] = df['Timestamp']
    min_timestamp = df['arrived_at'].min()
    df['arrived_at'] = df['arrived_at'] - min_timestamp
    
    # Map BurstGPT columns to Vidur format
    df['num_prefill_tokens'] = df['Request tokens']
    df['num_decode_tokens'] = df['Response tokens']
    
    # Ensure minimum token counts
    df['num_prefill_tokens'] = df['num_prefill_tokens'].clip(lower=1)
    df['num_decode_tokens'] = df['num_decode_tokens'].clip(lower=1)
    
    # Add block_size column (constant)
    df['block_size'] = block_size
    
    # Generate or set session_id
    if session_id_column and session_id_column in df.columns:
        df['session_id'] = df[session_id_column]
    else:
        # Assign sequential session IDs (each request is its own session)
        df['session_id'] = range(len(df))
    
    # Generate block_hash_ids if requested
    if generate_block_hashes:
        print("Generating block hash IDs...")
        block_hash_counter = 0
        block_hash_ids_list = []
        
        for idx, row in df.iterrows():
            num_prefill = int(row['num_prefill_tokens'])
            num_decode = int(row['num_decode_tokens'])
            total_tokens = num_prefill + num_decode
            num_blocks = (total_tokens + block_size - 1) // block_size
            
            # Generate sequential block IDs
            block_ids = list(range(block_hash_counter, block_hash_counter + num_blocks))
            block_hash_ids_list.append(json.dumps(block_ids))
            
            block_hash_counter += num_blocks
            
            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1:,} rows...")
        
        df['block_hash_ids'] = block_hash_ids_list
    else:
        df['block_hash_ids'] = None
    
    # Select and reorder columns for output
    output_columns = [
        'arrived_at',
        'num_prefill_tokens',
        'num_decode_tokens',
        # 'block_hash_ids',
        # 'block_size',
        'session_id'
    ]
    
    output_df = df[output_columns]
    
    # Save to output file
    print(f"\nSaving converted trace to: {output_file}")
    output_df.to_csv(output_file, index=False)
    
    # Print statistics
    print(f"\nConversion complete!")
    print(f"Output shape: {output_df.shape}")
    print(f"\nStatistics:")
    print(f"  Total requests: {len(output_df):,}")
    print(f"  Time range: {output_df['arrived_at'].min():.2f}s - {output_df['arrived_at'].max():.2f}s")
    print(f"  Duration: {output_df['arrived_at'].max() - output_df['arrived_at'].min():.2f}s")
    print(f"  Prefill tokens - mean: {output_df['num_prefill_tokens'].mean():.1f}, median: {output_df['num_prefill_tokens'].median():.1f}")
    print(f"  Decode tokens - mean: {output_df['num_decode_tokens'].mean():.1f}, median: {output_df['num_decode_tokens'].median():.1f}")
    
    # Preview output
    print(f"\nFirst 3 rows of output:")
    print(output_df.head(3).to_string())


def main():
    parser = argparse.ArgumentParser(
        description='Convert BurstGPT CSV format to Vidur trace format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input BurstGPT CSV file path'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output trace CSV file path'
    )
    
    parser.add_argument(
        '--block-size', '-b',
        type=int,
        default=16,
        help='Block size for prefix caching (default: 16, only supported value in Vidur)'
    )
    
    parser.add_argument(
        '--generate-block-hashes',
        action='store_true',
        default=False,
        help='Generate block_hash_ids for prefix caching (default: False)'
    )
    
    parser.add_argument(
        '--no-block-hashes',
        dest='generate_block_hashes',
        action='store_false',
        help='Do not generate block_hash_ids'
    )
    
    parser.add_argument(
        '--session-id-column',
        type=str,
        default=None,
        help='Column name in BurstGPT CSV to use as session_id (default: sequential IDs)'
    )
    
    args = parser.parse_args()
    
    # Validate block size
    if args.block_size != 16:
        print(f"Warning: Vidur only supports block_size=16, but you specified {args.block_size}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert
    convert_burstgpt_to_trace(
        input_file=args.input,
        output_file=args.output,
        block_size=args.block_size,
        generate_block_hashes=args.generate_block_hashes,
        session_id_column=args.session_id_column,
    )


if __name__ == '__main__':
    main()
