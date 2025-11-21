#!/usr/bin/env python3
"""
Plot scatter plots from Vidur simulator output CSV files.

This script can plot various scatter plots from the simulator output including:
- E2E time vs Request ID
- TTFT vs Request ID
- Prefill tokens vs Decode tokens
- Custom X vs Y columns

Usage:
    python plot_scatter.py <csv_file> [options]
    
Examples:
    # Plot E2E time scatter
    python plot_scatter.py request_e2e_time_scatter.csv
    
    # Plot TTFT scatter
    python plot_scatter.py request_ttft_scatter.csv --ylabel "TTFT (seconds)"
    
    # Custom columns
    python plot_scatter.py request_metrics.csv --x arrived_at --y e2e_time
    
    # Multiple files comparison
    python plot_scatter.py file1.csv file2.csv --labels "Config1" "Config2"
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional


def plot_scatter(
    csv_files: List[str],
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    output: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    labels: Optional[List[str]] = None,
    alpha: float = 0.5,
    figsize: tuple = (12, 6),
    style: str = 'scatter',
    show_stats: bool = True,
):
    """
    Create scatter plot from CSV file(s).
    
    Args:
        csv_files: List of CSV file paths
        x_column: Column name for X axis (auto-detected if None)
        y_column: Column name for Y axis (auto-detected if None)
        output: Output file path (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        labels: Legend labels for multiple files
        alpha: Point transparency (0-1)
        figsize: Figure size (width, height)
        style: 'scatter' or 'line'
        show_stats: Show statistics in legend
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, csv_file in enumerate(csv_files):
        print(f"[info] Loading: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Auto-detect columns if not specified
        if x_column is None:
            # Try common X column names
            possible_x = ['Request Id', 'request_id', 'arrived_at', 'index']
            x_col = None
            for col in possible_x:
                if col in df.columns:
                    x_col = col
                    break
            if x_col is None:
                x_col = df.columns[0]
        else:
            x_col = x_column
            
        if y_column is None:
            # Use second column as Y
            y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        else:
            y_col = y_column
        
        print(f"[info] Plotting: X={x_col}, Y={y_col}")
        
        # Get data
        x_data = df[x_col]
        y_data = df[y_col]
        
        # Calculate statistics
        mean_y = y_data.mean()
        median_y = y_data.median()
        p95_y = y_data.quantile(0.95)
        p99_y = y_data.quantile(0.99)
        
        # Create label
        if labels and idx < len(labels):
            label = labels[idx]
        else:
            label = Path(csv_file).stem
        
        if show_stats:
            label += f'\n(mean={mean_y:.2f}, p95={p95_y:.2f}, p99={p99_y:.2f})'
        
        # Plot
        if style == 'scatter':
            ax.scatter(
                x_data, 
                y_data,
                alpha=alpha,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                s=20,
                label=label,
                edgecolors='none'
            )
        elif style == 'line':
            ax.plot(
                x_data,
                y_data,
                alpha=0.7,
                color=colors[idx % len(colors)],
                linewidth=1,
                label=label
            )
        ax.set_xlim(left=15000,right=18000)
        ax.set_ylim(bottom=0,top=120)
        
        print(f"[stats] {Path(csv_file).name}:")
        print(f"  Points: {len(y_data):,}")
        print(f"  Mean: {mean_y:.2f}")
        print(f"  Median: {median_y:.2f}")
        print(f"  P95: {p95_y:.2f}")
        print(f"  P99: {p99_y:.2f}")
        print(f"  Min: {y_data.min():.2f}")
        print(f"  Max: {y_data.max():.2f}")
        print()
    
    # Set labels
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    else:
        ax.set_xlabel(x_col, fontsize=12)
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    else:
        ax.set_ylabel(y_col, fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'{y_col} vs {x_col}', fontsize=14, fontweight='bold')
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    if len(csv_files) > 1 or show_stats:
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save or show
    if output:
        print(f"[info] Saving to: {output}")
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"[done] Plot saved!")
    else:
        print("[info] Displaying plot...")
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot scatter plots from Vidur simulator CSV output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file with auto-detected columns
  python plot_scatter.py request_e2e_time_scatter.csv
  
  # Specify output file
  python plot_scatter.py request_e2e_time_scatter.csv --output e2e_plot.png
  
  # Custom columns
  python plot_scatter.py request_metrics.csv --x arrived_at --y e2e_time
  
  # Multiple files comparison
  python plot_scatter.py config1/request_e2e_time_scatter.csv config2/request_e2e_time_scatter.csv \\
      --labels "Baseline" "Optimized" --output comparison.png
  
  # Line plot instead of scatter
  python plot_scatter.py request_ttft_scatter.csv --style line
        """
    )
    
    parser.add_argument(
        'csv_files',
        nargs='+',
        help='CSV file(s) to plot'
    )
    parser.add_argument(
        '--x',
        dest='x_column',
        help='Column name for X-axis (auto-detected if not specified)'
    )
    parser.add_argument(
        '--y',
        dest='y_column',
        help='Column name for Y-axis (auto-detected if not specified)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file path (e.g., plot.png). If not specified, plot is displayed.'
    )
    parser.add_argument(
        '--title',
        help='Plot title'
    )
    parser.add_argument(
        '--xlabel',
        help='X-axis label'
    )
    parser.add_argument(
        '--ylabel',
        help='Y-axis label'
    )
    parser.add_argument(
        '--labels',
        nargs='+',
        help='Legend labels for multiple files'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Point transparency (0-1, default: 0.5)'
    )
    parser.add_argument(
        '--figsize',
        nargs=2,
        type=float,
        default=[12, 6],
        help='Figure size: width height (default: 12 6)'
    )
    parser.add_argument(
        '--style',
        choices=['scatter', 'line'],
        default='scatter',
        help='Plot style (default: scatter)'
    )
    parser.add_argument(
        '--no-stats',
        dest='show_stats',
        action='store_false',
        help='Hide statistics in legend'
    )
    
    args = parser.parse_args()
    
    plot_scatter(
        csv_files=args.csv_files,
        x_column=args.x_column,
        y_column=args.y_column,
        output=args.output,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        labels=args.labels,
        alpha=args.alpha,
        figsize=tuple(args.figsize),
        style=args.style,
        show_stats=args.show_stats,
    )


if __name__ == '__main__':
    main()
