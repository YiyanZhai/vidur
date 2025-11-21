#!/usr/bin/env python3
"""
Convert ShareGPT Vicuna dataset to Mooncake conversation trace format.

Output CSV format:
- arrived_at: timestamp (float, in seconds)
- num_prefill_tokens: number of tokens in the prompt
- num_decode_tokens: number of tokens in the response
- block_hash_ids: list of block hash IDs (for prefix caching simulation)
- block_size: block size for KV cache (default 16)
- session_id: conversation session ID

Requires: datasets, tiktoken, pandas
"""

import argparse
import csv
import json
import random
from typing import List, Tuple, Optional

from datasets import load_dataset


def try_load_tiktoken():
    """Load tiktoken encoder for token counting."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"Warning: tiktoken not available ({e}). Install with: pip install tiktoken")
        return None


def count_tokens(text: str, encoder) -> int:
    """Count tokens in text. Falls back to word count if tiktoken unavailable."""
    if encoder is not None:
        return len(encoder.encode(text))
    else:
        # Fallback: rough approximation
        return len(text.strip().split())


def generate_block_hash_ids(
    num_prefill_tokens: int,
    num_decode_tokens: int,
    block_size: int,
    session_id: int,
    turn_id: int,
    previous_context_blocks: int = 0,
    include_previous_blocks: bool = False,
) -> List[int]:
    """
    Generate block hash IDs for prefix caching simulation.
    
    Args:
        include_previous_blocks: If True, include all previous context blocks in the list
                                (multi-turn representation). If False, only include current
                                request's blocks (default behavior).
    
    The block hash IDs represent:
    - When include_previous_blocks=False (default):
      * Only blocks for THIS REQUEST (prompt + response tokens)
      * Block IDs are sequential starting after previous context
      * Prefix sharing is encoded in block ID values
    
    - When include_previous_blocks=True (multi-turn mode):
      * All blocks from conversation start up to and including current turn
      * Simulates accumulated KV cache in multi-turn conversations
      * Previous context blocks + new blocks from current turn
    
    Only full blocks (exactly block_size tokens) are included.
    Partial blocks at the end are excluded.
    """
    # Total tokens in THIS request = prompt + response
    total_tokens = num_prefill_tokens + num_decode_tokens
    # Only count full blocks - exclude partial last block
    num_full_blocks = total_tokens // block_size
    
    # Generate sequential block IDs based on session
    base_offset = session_id * 10000  # Each session gets a different range
    
    if include_previous_blocks and turn_id > 0:
        # Multi-turn mode: include all previous context blocks + new blocks
        # Previous context blocks (from earlier turns in the conversation)
        shared_prefix = list(range(base_offset, base_offset + previous_context_blocks))
        # New blocks for current turn
        new_block_start = base_offset + previous_context_blocks
        new_blocks = list(range(new_block_start, new_block_start + num_full_blocks))
        block_ids = shared_prefix + new_blocks
    else:
        # Default mode: only blocks for this request
        block_start = base_offset + previous_context_blocks
        block_ids = list(range(block_start, block_start + num_full_blocks))
    
    return block_ids


def extract_conversations_from_sharegpt(
    dataset,
    encoder,
    max_conversations: Optional[int] = None,
    block_size: int = 16,
    start_time: float = 0.0,
    time_interval_mean: float = 1.0,
    time_interval_std: float = 0.5,
    seed: int = 42,
    max_decode_tokens: Optional[int] = None,
    min_prefill_tokens: int = 1,
    streaming: bool = False,
    multi_turn_mode: bool = False,
) -> List[dict]:
    """
    Extract conversations from ShareGPT Vicuna dataset.
    
    Each conversation may have multiple turns. We'll create a trace entry for each turn.
    
    Args:
        multi_turn_mode: If True, subsequent turns include accumulated context in num_prefill_tokens
                        and block_hash_ids includes all previous blocks. If False (default), each
                        turn is treated independently.
    """
    random.seed(seed)
    records = []
    session_id = 0
    current_time = start_time
    
    conv_count = 0
    
    # Track context accumulation for multi-turn conversations
    session_context_blocks = {}  # session_id -> total blocks so far
    session_context_tokens = {}  # session_id -> total tokens so far (for multi-turn mode)
    
    if streaming:
        iterator = dataset
    else:
        iterator = dataset
    
    for example in iterator:
        conversations = example.get("conversations", [])
        # print(f"Processing session {session_id} with {len(conversations)} turns")
        
        if not conversations:
            continue
        
        # Process each turn in the conversation
        turn_id = 0
        for i in range(0, len(conversations) - 1, 2):  # Process pairs (human, gpt))
            human_turn = conversations[i]
            gpt_turn = conversations[i + 1] if i + 1 < len(conversations) else None
            
            # Verify this is a human-gpt pair
            if human_turn.get("from", "").lower() not in ("human", "user"):
                continue
            if gpt_turn is None or gpt_turn.get("from", "").lower() not in ("gpt", "assistant"):
                continue
            
            prompt_text = human_turn.get("value", "")
            response_text = gpt_turn.get("value", "")
            # print(f" Turn {turn_id}: human from '{human_turn.get('value', '')}', gpt from '{gpt_turn.get('value', '')}'")
            
            if not prompt_text or not response_text:
                continue
            
            # Count tokens for this turn
            num_prefill_tokens_current = count_tokens(prompt_text, encoder)
            num_decode_tokens = count_tokens(response_text, encoder)
            
            # Apply filters
            if num_prefill_tokens_current < min_prefill_tokens:
                continue
            if max_decode_tokens and num_decode_tokens > max_decode_tokens:
                num_decode_tokens = max_decode_tokens
            
            # Get previous context for this session
            previous_blocks = session_context_blocks.get(session_id, 0)
            previous_tokens = session_context_tokens.get(session_id, 0)
            
            # In multi-turn mode, prefill includes all previous context
            if multi_turn_mode and turn_id > 0:
                # Prefill = all previous conversation + current prompt
                num_prefill_tokens = previous_tokens + num_prefill_tokens_current
            else:
                # Independent mode: only current prompt
                num_prefill_tokens = num_prefill_tokens_current
            
            # Generate block hash IDs
            # In multi-turn mode, we need to set previous_blocks to 0 since
            # num_prefill_tokens already includes all accumulated tokens
            prev_blocks_arg = 0 if multi_turn_mode else previous_blocks
            
            block_hash_ids = generate_block_hash_ids(
                num_prefill_tokens,
                num_decode_tokens,
                block_size,
                session_id,
                turn_id,
                prev_blocks_arg,
                include_previous_blocks=multi_turn_mode,
            )
            
            # Update context: add tokens/blocks from current turn (prompt + response)
            # Only count full blocks for the next turn's prefix
            current_turn_tokens = num_prefill_tokens_current + num_decode_tokens
            current_turn_full_blocks = current_turn_tokens // block_size
            session_context_blocks[session_id] = previous_blocks + current_turn_full_blocks
            session_context_tokens[session_id] = previous_tokens + current_turn_tokens
            
            # Add timestamp jitter for realism
            time_jitter = max(0.0, random.gauss(time_interval_mean, time_interval_std))
            current_time += time_jitter
            
            record = {
                "arrived_at": round(current_time, 2),
                "num_prefill_tokens": num_prefill_tokens,
                "num_decode_tokens": num_decode_tokens,
                "block_hash_ids": json.dumps(block_hash_ids),
                "block_size": block_size,
                "session_id": session_id,
            }
            records.append(record)
            turn_id += 1
        
        # Move to next conversation
        if turn_id > 0:  # Only count if we extracted at least one turn
            session_id += 1
            conv_count += 1
        
        if max_conversations and conv_count >= max_conversations:
            break
    
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Convert ShareGPT Vicuna dataset to Mooncake trace format"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sharegpt_mooncake_trace.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--max_conversations",
        type=int,
        default=None,
        help="Maximum number of conversations to process (None for all)",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=16,
        help="Block size for KV cache",
    )
    parser.add_argument(
        "--start_time",
        type=float,
        default=0.0,
        help="Starting timestamp",
    )
    parser.add_argument(
        "--time_interval_mean",
        type=float,
        default=1.0,
        help="Mean time interval between requests (seconds)",
    )
    parser.add_argument(
        "--time_interval_std",
        type=float,
        default=0.5,
        help="Standard deviation of time intervals",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for timestamp generation",
    )
    parser.add_argument(
        "--max_decode_tokens",
        type=int,
        default=None,
        help="Cap maximum decode tokens per turn",
    )
    parser.add_argument(
        "--min_prefill_tokens",
        type=int,
        default=1,
        help="Minimum prefill tokens to include",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for loading dataset",
    )
    parser.add_argument(
        "--multi_turn",
        action="store_true",
        help="Enable multi-turn mode: subsequent turns include accumulated context in "
             "num_prefill_tokens and block_hash_ids includes all previous blocks. "
             "This simulates true multi-turn conversations with prefix caching.",
    )
    
    args = parser.parse_args()
    
    # Load tiktoken encoder
    print("[info] Loading tiktoken encoder...")
    encoder = try_load_tiktoken()
    if encoder is None:
        print("[warning] Using word-based token counting (less accurate)")
    
    # Load ShareGPT Vicuna dataset
    print("[info] Loading ShareGPT Vicuna dataset...")
    dataset = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        split="train",
        data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
        streaming=args.streaming,
    )
    
    if not args.streaming:
        print(f"[info] Dataset loaded: {len(dataset)} conversations")
    else:
        print("[info] Dataset loaded in streaming mode")
    
    # Extract and convert conversations
    mode_desc = "multi-turn mode" if args.multi_turn else "independent turn mode"
    print(f"[info] Extracting conversations ({mode_desc})...")
    records = extract_conversations_from_sharegpt(
        dataset,
        encoder,
        max_conversations=args.max_conversations,
        block_size=args.block_size,
        start_time=args.start_time,
        time_interval_mean=args.time_interval_mean,
        time_interval_std=args.time_interval_std,
        seed=args.seed,
        max_decode_tokens=args.max_decode_tokens,
        min_prefill_tokens=args.min_prefill_tokens,
        streaming=args.streaming,
        multi_turn_mode=args.multi_turn,
    )
    
    print(f"[info] Extracted {len(records)} turns from conversations")
    
    # Write to CSV
    print(f"[info] Writing to {args.output}...")
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "arrived_at",
            "num_prefill_tokens",
            "num_decode_tokens",
            "block_hash_ids",
            "block_size",
            "session_id",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    
    print(f"[done] Wrote {len(records)} records to {args.output}")
    
    # Print statistics
    if records:
        avg_prefill = sum(r["num_prefill_tokens"] for r in records) / len(records)
        avg_decode = sum(r["num_decode_tokens"] for r in records) / len(records)
        total_time = records[-1]["arrived_at"] - records[0]["arrived_at"]
        print(f"[stats] Average prefill tokens: {avg_prefill:.1f}")
        print(f"[stats] Average decode tokens: {avg_decode:.1f}")
        print(f"[stats] Total time span: {total_time:.2f} seconds")
        print(f"[stats] Average QPS: {len(records) / max(total_time, 1):.2f}")


if __name__ == "__main__":
    main()
