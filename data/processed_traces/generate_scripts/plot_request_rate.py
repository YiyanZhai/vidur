#!/usr/bin/env python3
"""
Plot request rate (requests per second) over time for trace files.

Usage:
    python plot_request_rate.py <trace_file.csv> [--window_size 10] [--output plot.png]
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_request_rate(
    trace_file: str,
    window_size: float = 10.0,
    output_file: str = None,
    show_plot: bool = True,
):
    """
    Plot request rate over time.
    
    Args:
        trace_file: Path to CSV trace file with 'arrived_at' column
        window_size: Time window in seconds for calculating rate
        output_file: Output file path for saving the plot (optional)
        show_plot: Whether to display the plot interactively
    """
    print(f"[info] Loading trace file: {trace_file}")
    df = pd.read_csv(trace_file)
    
    if 'arrived_at' not in df.columns:
        raise ValueError("Trace file must have 'arrived_at' column")
    
    # Sort by arrival time
    df = df.sort_values('arrived_at')
    
    # Get time range
    min_time = df['arrived_at'].min()
    max_time = df['arrived_at'].max()
    time_span = max_time - min_time
    
    print(f"[info] Total requests: {len(df)}")
    print(f"[info] Time range: {min_time:.2f}s to {max_time:.2f}s ({time_span:.2f}s)")
    print(f"[info] Average request rate: {len(df) / time_span:.2f} req/s")
    
    # Create time bins
    num_bins = int(np.ceil(time_span / window_size))
    bins = np.linspace(min_time, max_time, num_bins + 1)
    
    # Count requests per bin
    counts, bin_edges = np.histogram(df['arrived_at'], bins=bins)
    
    # Calculate rate (requests per second)
    rates = counts / window_size
    
    # Use bin centers for x-axis
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Request rate over time
    ax1.plot(bin_centers, rates, linewidth=1.5, color='#2E86AB')
    ax1.fill_between(bin_centers, rates, alpha=0.3, color='#2E86AB')
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_ylabel('Request Rate (req/s)', fontsize=11)
    ax1.set_title(f'Request Rate Over Time (window={window_size}s)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_rate = np.mean(rates)
    median_rate = np.median(rates)
    max_rate = np.max(rates)
    ax1.axhline(mean_rate, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Mean: {mean_rate:.2f} req/s')
    ax1.axhline(median_rate, color='orange', linestyle='--', linewidth=1, alpha=0.7, label=f'Median: {median_rate:.2f} req/s')
    ax1.legend(loc='upper right', fontsize=9)
    
    # Plot 2: Cumulative requests over time
    cumulative_times = df['arrived_at'].values
    cumulative_counts = np.arange(1, len(df) + 1)
    
    ax2.plot(cumulative_times, cumulative_counts, linewidth=1.5, color='#A23B72')
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('Cumulative Requests', fontsize=11)
    ax2.set_title('Cumulative Requests Over Time', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        print(f"[info] Saving plot to: {output_file}")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show plot
    if show_plot:
        print("[info] Displaying plot...")
        plt.show()
    else:
        plt.close()
    
    # Print statistics
    print("\n[stats] Request Rate Statistics:")
    print(f"  Mean rate: {mean_rate:.2f} req/s")
    print(f"  Median rate: {median_rate:.2f} req/s")
    print(f"  Max rate: {max_rate:.2f} req/s")
    print(f"  Min rate: {np.min(rates):.2f} req/s")
    print(f"  Std dev: {np.std(rates):.2f} req/s")
    print(f"  95th percentile: {np.percentile(rates, 95):.2f} req/s")
    print(f"  99th percentile: {np.percentile(rates, 99):.2f} req/s")


def main():
    parser = argparse.ArgumentParser(
        description="Plot request rate over time for trace files"
    )
    parser.add_argument(
        "trace_file",
        type=str,
        help="Path to CSV trace file",
    )
    parser.add_argument(
        "--window_size",
        type=float,
        default=10.0,
        help="Time window in seconds for calculating rate (default: 10.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for saving the plot (e.g., plot.png)",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Don't display the plot interactively (useful when saving only)",
    )
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not specified but no_show is set
    output_file = args.output
    if args.no_show and output_file is None:
        trace_path = Path(args.trace_file)
        output_file = trace_path.parent / f"{trace_path.stem}_rate_plot.png"
    
    plot_request_rate(
        args.trace_file,
        window_size=args.window_size,
        output_file=output_file,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    main()
