"""
Track estimated TTFT for each request to enable comparison with actual TTFT.
"""

from typing import Dict, Optional
import pandas as pd
from pathlib import Path

from vidur import logger


class TTFTEstimateTracker:
    """
    Tracks estimated TTFT for each request in the queue.
    Allows exporting estimates for comparison with actual TTFT from another run.
    """

    def __init__(self):
        """Initialize the TTFT estimate tracker."""
        self._estimates: Dict[int, Dict[str, float]] = {}
        # Stores: request_id -> {"estimated_ttft": float, "current_time": float, "deadline": float, ...}

    def record_estimate(
        self,
        request_id: int,
        estimated_ttft: float,
        current_time: float,
        deadline: Optional[float] = None,
        remaining_prefill_tokens: int = 0,
        queue_position: int = 0,
        ahead_prefill_tokens: int = 0,
    ) -> None:
        """
        Record an estimated TTFT for a request.

        Args:
            request_id: Unique request identifier
            estimated_ttft: Estimated time-to-first-token (seconds)
            current_time: Current simulation time (seconds)
            deadline: Prefill deadline time (seconds), if available
            remaining_prefill_tokens: Number of prefill tokens remaining
            queue_position: Position in the waiting queue (0-indexed)
            ahead_prefill_tokens: Total prefill tokens ahead in queue
        """
        if request_id in self._estimates:
            return
        self._estimates[request_id] = {
            "estimated_ttft": estimated_ttft,
            "current_time": current_time,
            "estimated_completion_time": current_time + estimated_ttft,
            "deadline": deadline if deadline is not None else float("inf"),
            "time_until_deadline": (deadline - current_time) if deadline is not None else float("inf"),
            "slack": (deadline - current_time - estimated_ttft) if deadline is not None else float("inf"),
            "remaining_prefill_tokens": remaining_prefill_tokens,
            "queue_position": queue_position,
            "ahead_prefill_tokens": ahead_prefill_tokens,
            "is_violation": (estimated_ttft > (deadline - current_time)) if deadline is not None else False,
        }

    def get_estimate(self, request_id: int) -> Optional[Dict[str, float]]:
        """
        Get the stored estimate for a request.

        Args:
            request_id: Request identifier

        Returns:
            Dictionary with estimate details, or None if not found
        """
        return self._estimates.get(request_id)

    def has_estimate(self, request_id: int) -> bool:
        """Check if we have an estimate for a request."""
        return request_id in self._estimates

    def clear_estimate(self, request_id: int) -> None:
        """Remove an estimate (e.g., when request completes or is outsourced)."""
        if request_id in self._estimates:
            del self._estimates[request_id]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export all estimates to a pandas DataFrame.

        Returns:
            DataFrame with columns: request_id, estimated_ttft, current_time,
                                   deadline, is_violation, etc.
        """
        if not self._estimates:
            return pd.DataFrame()

        records = []
        for req_id, data in self._estimates.items():
            record = {"request_id": req_id, **data}
            records.append(record)

        df = pd.DataFrame(records)
        # Sort by request_id for easier comparison
        df = df.sort_values("request_id").reset_index(drop=True)
        return df

    def save_to_csv(self, filepath: str) -> None:
        """
        Save all estimates to a CSV file.

        Args:
            filepath: Output CSV file path
        """
        df = self.to_dataframe()
        if df.empty:
            logger.warning("No TTFT estimates to save")
            return

        # Create parent directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(filepath, index=False)
        # logger.info(f"Saved {len(df)} TTFT estimates to {filepath}")

    def get_summary_stats(self) -> Dict[str, float]:
        """
        Get summary statistics about the tracked estimates.

        Returns:
            Dictionary with summary stats (mean, median, violations, etc.)
        """
        if not self._estimates:
            return {}

        df = self.to_dataframe()
        stats = {
            "num_requests": len(df),
            "mean_estimated_ttft": df["estimated_ttft"].mean(),
            "median_estimated_ttft": df["estimated_ttft"].median(),
            "max_estimated_ttft": df["estimated_ttft"].max(),
            "min_estimated_ttft": df["estimated_ttft"].min(),
            "num_violations": df["is_violation"].sum(),
            "violation_rate": df["is_violation"].mean(),
            "mean_slack": df[df["slack"] != float("inf")]["slack"].mean(),
        }
        return stats

    def clear_all(self) -> None:
        """Clear all tracked estimates."""
        self._estimates.clear()

    def __len__(self) -> int:
        """Return the number of tracked estimates."""
        return len(self._estimates)

    def __repr__(self) -> str:
        """String representation."""
        return f"TTFTEstimateTracker(num_estimates={len(self._estimates)})"
