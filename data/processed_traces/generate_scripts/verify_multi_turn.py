#!/usr/bin/env python3
import pandas as pd
import json

print("=" * 80)
print("MULTI-TURN MODE VERIFICATION")
print("=" * 80)

df = pd.read_csv('test_multi_turn_v2.csv')
print(f'\nTotal records: {len(df)}')
print('\nDetailed breakdown (first session):')

session_0 = df[df['session_id'] == 0]
for idx, (i, row) in enumerate(session_0.iterrows()):
    blocks = json.loads(row['block_hash_ids'])
    prefill = row['num_prefill_tokens']
    decode = row['num_decode_tokens']
    total = prefill + decode
    last_block = total - (16 * len(blocks))
    
    print(f'\nTurn {idx} (session {row["session_id"]}):')
    print(f'  Prefill: {prefill} tokens | Decode: {decode} tokens | Total: {total} tokens')
    print(f'  Blocks: {len(blocks)} | Last block size: {last_block} | Valid: {0 <= last_block < 16}')
    if len(blocks) <= 10:
        print(f'  Block IDs: {blocks}')
    else:
        print(f'  Block IDs: [{blocks[0]}, {blocks[1]}, {blocks[2]}, ..., {blocks[-3]}, {blocks[-2]}, {blocks[-1]}] (range: {blocks[0]} to {blocks[-1]})')
    
    if idx > 0:
        prev_blocks = json.loads(df.iloc[i-1]['block_hash_ids'])
        shared = [b for b in prev_blocks if b in blocks]
        print(f'  Shared blocks with previous turn: {len(shared)}/{len(prev_blocks)} blocks')
        if len(shared) == len(prev_blocks):
            print(f'  ✓ All previous blocks included (correct prefix caching!)')

