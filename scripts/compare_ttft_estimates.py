"""
Script to compare estimated TTFT with actual TTFT from simulation runs.

Usage:
    python scripts/compare_ttft_estimates.py \
        --run_dir simulator_output/vanilla_1024

This script loads:
1. Estimated TTFT from ttft_estimates_replica_0.csv
2. Actual TTFT from request_metrics.csv
3. Output file name is ttft_comparison_{run_dir.split('/')[-1]}.png)

And produces comparison plots and statistics.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_estimated_ttft(estimates_csv: str) -> pd.DataFrame:
    """Load estimated TTFT data."""
    df = pd.read_csv(estimates_csv)
    print(f"Loaded {len(df)} estimated TTFT records")
    print(f"Columns: {df.columns.tolist()}")
    return df


def load_actual_ttft(metrics_csv: str) -> pd.DataFrame:
    """
    Load actual TTFT from request metrics.
    TTFT = prefill_completed_at - queued_at (in seconds)
    """
    df = pd.read_csv(metrics_csv)
    df['actual_ttft'] = df['prefill_e2e_time']
    
    # # Calculate actual TTFT
    # if 'prefill_completed_at' in df.columns and 'queued_at' in df.columns:
    #     df['actual_ttft'] = df['prefill_completed_at'] - df['queued_at']
    # elif 'prefill_completed_at' in df.columns and 'arrived_at' in df.columns:
    #     df['actual_ttft'] = df['prefill_completed_at'] - df['arrived_at']
    # else:
    #     raise ValueError(f"Cannot calculate TTFT from columns: {df.columns.tolist()}")
    
    # Rename 'id' to 'request_id' for consistency
    if 'Request Id' in df.columns:
        df = df.rename(columns={'Request Id': 'request_id'})
    
    print(f"Loaded {len(df)} actual request records")
    return df[['request_id', 'actual_ttft']]


def merge_and_compare(estimates_df: pd.DataFrame, actual_df: pd.DataFrame) -> pd.DataFrame:
    """Merge estimated and actual TTFT by request_id."""
    merged = pd.merge(estimates_df, actual_df, on='request_id', how='inner')
    print(f"Matched {len(merged)} requests between estimated and actual data")
    
    # Calculate errors
    merged['absolute_error'] = merged['actual_ttft'] - merged['estimated_ttft']
    merged['relative_error'] = (merged['actual_ttft'] - merged['estimated_ttft']) / merged['actual_ttft']
    merged['percent_error'] = merged['relative_error'] * 100
    
    return merged


def print_statistics(comparison_df: pd.DataFrame):
    """Print comparison statistics."""
    print("\n" + "="*60)
    print("TTFT COMPARISON STATISTICS")
    print("="*60)
    
    print(f"\nNumber of requests: {len(comparison_df)}")
    
    print(f"\nEstimated TTFT:")
    print(f"  Mean: {comparison_df['estimated_ttft'].mean():.4f}s")
    print(f"  Median: {comparison_df['estimated_ttft'].median():.4f}s")
    print(f"  Std: {comparison_df['estimated_ttft'].std():.4f}s")
    
    print(f"\nActual TTFT:")
    print(f"  Mean: {comparison_df['actual_ttft'].mean():.4f}s")
    print(f"  Median: {comparison_df['actual_ttft'].median():.4f}s")
    print(f"  Std: {comparison_df['actual_ttft'].std():.4f}s")
    
    print(f"\nAbsolute Error (actual - estimated):")
    print(f"  Mean: {comparison_df['absolute_error'].mean():.4f}s")
    print(f"  Median: {comparison_df['absolute_error'].median():.4f}s")
    print(f"  MAE: {comparison_df['absolute_error'].abs().mean():.4f}s")
    print(f"  RMSE: {np.sqrt((comparison_df['absolute_error']**2).mean()):.4f}s")
    
    print(f"\nPercent Error:")
    print(f"  Mean: {comparison_df['percent_error'].mean():.2f}%")
    print(f"  Median: {comparison_df['percent_error'].median():.2f}%")
    print(f"  MAPE: {comparison_df['percent_error'].abs().mean():.2f}%")
    
    # Check underestimation vs overestimation
    underestimated = (comparison_df['absolute_error'] > 0).sum()
    overestimated = (comparison_df['absolute_error'] < 0).sum()
    print(f"\nEstimation Bias:")
    print(f"  Underestimated (actual > estimated): {underestimated} ({underestimated/len(comparison_df)*100:.1f}%)")
    print(f"  Overestimated (actual < estimated): {overestimated} ({overestimated/len(comparison_df)*100:.1f}%)")
    
    print("="*60 + "\n")


def plot_comparison(comparison_df: pd.DataFrame, output_path: str):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Scatter plot: Estimated vs Actual
    ax = axes[0, 0]
    ax.scatter(comparison_df['estimated_ttft'], comparison_df['actual_ttft'], 
               alpha=0.5, s=20)
    
    # Add diagonal line (perfect prediction)
    min_val = min(comparison_df['estimated_ttft'].min(), comparison_df['actual_ttft'].min())
    max_val = max(comparison_df['estimated_ttft'].max(), comparison_df['actual_ttft'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    ax.set_xlabel('Estimated TTFT (seconds)')
    ax.set_ylabel('Actual TTFT (seconds)')
    ax.set_title('Estimated vs Actual TTFT')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Error distribution
    ax = axes[0, 1]
    ax.hist(comparison_df['absolute_error'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel('Absolute Error (seconds)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution (Actual - Estimated)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Percent error distribution
    ax = axes[1, 0]
    ax.hist(comparison_df['percent_error'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel('Percent Error (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Percent Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Error vs Queue Position
    ax = axes[1, 1]
    if 'queue_position' in comparison_df.columns:
        scatter = ax.scatter(comparison_df['queue_position'], 
                           comparison_df['absolute_error'], 
                           c=comparison_df['estimated_ttft'],
                           cmap='viridis', alpha=0.6, s=20)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Queue Position')
        ax.set_ylabel('Absolute Error (seconds)')
        ax.set_title('Error vs Queue Position')
        plt.colorbar(scatter, ax=ax, label='Estimated TTFT (s)')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Queue position data not available', 
               ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare estimated vs actual TTFT from simulation runs'
    )
    parser.add_argument('--run_dir', required=True,
                       help='Path to the simulation run directory (e.g., simulator_output/vanilla_1024)')
    
    args = parser.parse_args()
    
    # Construct file paths
    run_dir = Path(args.run_dir)
    estimates_path = run_dir / 'ttft_estimates_replica_0.csv'
    actual_path = run_dir / 'request_metrics.csv'
    
    # Construct output filename based on run_dir
    run_name = run_dir.name  # Gets the last part of the path (e.g., 'vanilla_1024')
    output_path = f'experiment_res/ttft_comparison_{run_name}.png'
    
    # Check if input files exist
    if not estimates_path.exists():
        print(f"ERROR: Estimates file not found: {estimates_path}")
        return
    if not actual_path.exists():
        print(f"ERROR: Actual metrics file not found: {actual_path}")
        return
    
    print(f"Loading data from: {run_dir}")
    print(f"  Estimates: {estimates_path}")
    print(f"  Actual: {actual_path}")
    print(f"  Output: {output_path}")
    print()
    
    # Load data
    estimates_df = load_estimated_ttft(str(estimates_path))
    actual_df = load_actual_ttft(str(actual_path))
    
    # Merge and compare
    comparison_df = merge_and_compare(estimates_df, actual_df)
    
    if len(comparison_df) == 0:
        print("ERROR: No matching requests found between estimated and actual data!")
        return
    
    # Print statistics
    print_statistics(comparison_df)
    
    # Save comparison data
    output_csv = Path(output_path).with_suffix('.csv')
    comparison_df.to_csv(output_csv, index=False)
    print(f"Saved comparison data to: {output_csv}")
    
    # Create plots
    plot_comparison(comparison_df, output_path)


if __name__ == '__main__':
    main()
