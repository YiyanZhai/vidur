import logging
from typing import List

import pandas as pd

from vidur.config import BurstGPTRequestGeneratorConfig
from vidur.entities import Request
from vidur.request_generator.base_request_generator import BaseRequestGenerator

logger = logging.getLogger(__name__)


class BurstGPTRequestGenerator(BaseRequestGenerator):
    """
    Reads a BurstGPT trace CSV file containing request arrival time, request/response tokens
    to generate requests for simulation. The BurstGPT dataset format has columns:
    Timestamp, Model, Request tokens, Response tokens, Total tokens, Log Type
    """

    def __init__(self, config: BurstGPTRequestGeneratorConfig):
        super().__init__(config)

        # Load the BurstGPT CSV file
        self.trace_df = pd.read_csv(config.trace_file)
        self.trace_df = self.trace_df[self.trace_df['Log Type'] != 'Conversation log']
        self.trace_df['Timestamp'] = self.trace_df['Timestamp'] - self.trace_df['Timestamp'].iloc[0]
        self.trace_df = pd.DataFrame(
            self.trace_df.values.repeat(2, axis=0),
            columns=self.trace_df.columns
        ).reset_index(drop=True)
        
        logger.info(f"Loaded BurstGPT trace file {config.trace_file} with {len(self.trace_df)} requests")
        
        # Rename columns to match expected format
        self.trace_df = self.trace_df.rename(columns={
            'Timestamp': 'arrived_at',
            'Request tokens': 'num_prefill_tokens', 
            'Response tokens': 'num_decode_tokens'
        })
        
        # Apply scaling factors
        self.trace_df["num_prefill_tokens"] = (
            self.trace_df["num_prefill_tokens"] * config.prefill_scale_factor
        )
        self.trace_df["num_decode_tokens"] = (
            self.trace_df["num_decode_tokens"] * config.decode_scale_factor
        )

        # Convert to integers
        self.trace_df["num_prefill_tokens"] = self.trace_df["num_prefill_tokens"].astype(int)
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].astype(int)

        # Ensure at least one token for prefill and decode
        self.trace_df["num_prefill_tokens"] = self.trace_df["num_prefill_tokens"].clip(lower=1)
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].clip(lower=1)

        # Filter out failed requests (where response tokens might be 0 in original data)
        # This should already be handled by the clip(lower=1) above, but let's be explicit
        initial_count = len(self.trace_df)
        self.trace_df = self.trace_df[
            (self.trace_df["num_prefill_tokens"] > 0) & 
            (self.trace_df["num_decode_tokens"] > 0)
        ]
        final_count = len(self.trace_df)
        if initial_count != final_count:
            logger.info(f"Filtered out {initial_count - final_count} requests with zero tokens")

        # Ensure total tokens don't exceed max_tokens, adjust prefill if needed
        total_tokens = self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
        excess_tokens = (total_tokens - config.max_tokens).clip(lower=0)
        self.trace_df["num_prefill_tokens"] = (
            self.trace_df["num_prefill_tokens"] - excess_tokens
        ).clip(lower=1)  # Ensure prefill tokens stay >= 1
        
        # Additionally, ensure individual token counts stay within execution predictor limits
        # The default prediction_max_tokens_per_request is 4096, so we should stay below that
        prediction_limit = 4096  # Leave some margin below 4096
        self.trace_df["num_prefill_tokens"] = self.trace_df["num_prefill_tokens"].clip(upper=prediction_limit)
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].clip(upper=prediction_limit)
        
        # Re-apply the total token constraint after individual clamping
        total_tokens = self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
        excess_tokens = (total_tokens - config.max_tokens).clip(lower=0)
        self.trace_df["num_prefill_tokens"] = (
            self.trace_df["num_prefill_tokens"] - excess_tokens
        ).clip(lower=1)  # Ensure prefill tokens stay >= 1

        # Apply time scaling
        self.trace_df["arrived_at"] = self.trace_df["arrived_at"] * config.time_scale_factor
        
        # Sort by arrival time to ensure proper ordering
        self.trace_df = self.trace_df.sort_values('arrived_at').reset_index(drop=True)
        
        # Limit number of requests if specified
        if config.num_requests is not None and config.num_requests > 0:
            self.trace_df = self.trace_df.head(config.num_requests)
            logger.info(f"Limited to {len(self.trace_df)} requests as specified in config")

        # Log statistics
        logger.info(f"Final dataset has {len(self.trace_df)} requests")
        logger.info(f"Time range: {self.trace_df['arrived_at'].min():.2f} to {self.trace_df['arrived_at'].max():.2f} seconds")
        
        # Compute and log prompt/decode ratio statistics
        pd_ratio = self.trace_df["num_prefill_tokens"] / self.trace_df["num_decode_tokens"]
        logger.info(f"Prompt/decode token ratio stats:\n{pd_ratio.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])}")
        
        # Log token statistics
        logger.info(f"Prefill tokens stats:\n{self.trace_df['num_prefill_tokens'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])}")
        logger.info(f"Decode tokens stats:\n{self.trace_df['num_decode_tokens'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])}")

    def generate_requests(self) -> List[Request]:
        requests = []

        for _, row in self.trace_df.iterrows():
            request = Request(
                arrived_at=row["arrived_at"],
                num_prefill_tokens=row["num_prefill_tokens"],
                num_decode_tokens=row["num_decode_tokens"],
            )
            requests.append(request)

        logger.info(f"Generated {len(requests)} requests from BurstGPT trace")
        return requests
