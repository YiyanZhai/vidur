#!/usr/bin/env python3
"""
Compare scatter plots from multiple Vidur simulation outputs.

Usage:
    python compare_scatter.py <metric_name> <output_dir1> <output_dir2> ... [options]
    
Examples:
    # Compare E2E times
    python scripts/compare_scatter.py request_e2e_time_scatter \
        simulator_output/combined_vanilla_1/plots \
        simulator_output/combined_vanilla_2reps_1/plots \
        --labels "1" "2reps_1" \
        --output comparison_e2e.png
    
    # Compare TTFT
    python scripts/compare_scatter.py request_ttft_scatter \
        simulator_output/config1/plots \
        simulator_output/config2/plots \
        simulator_output/config3/plots \
        --labels "Config1" "Config2" "Config3"
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from plot_scatter import plot_scatter


def main():
    parser = argparse.ArgumentParser(
        description="Compare scatter plots from multiple simulation outputs"
    )
    
    parser.add_argument(
        'metric_name',
        help='Metric CSV filename (without .csv) or with .csv extension'
    )
    parser.add_argument(
        'output_dirs',
        nargs='+',
        help='Output directories containing plots/'
    )
    parser.add_argument(
        '--labels',
        nargs='+',
        help='Labels for each output directory'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file path'
    )
    parser.add_argument(
        '--title',
        help='Plot title (auto-generated if not specified)'
    )
    parser.add_argument(
        '--ylabel',
        help='Y-axis label'
    )
    parser.add_argument(
        '--style',
        choices=['scatter', 'line'],
        default='scatter',
        help='Plot style (default: scatter)'
    )
    
    args = parser.parse_args()
    
    # Add .csv extension if not present
    metric_name = args.metric_name
    if not metric_name.endswith('.csv'):
        metric_name += '.csv'
    
    # Find CSV files
    csv_files = []
    for output_dir in args.output_dirs:
        csv_path = Path(output_dir) / metric_name
        if csv_path.exists():
            csv_files.append(str(csv_path))
        else:
            print(f"[warning] File not found: {csv_path}")
    
    if not csv_files:
        print("[error] No CSV files found!")
        return 1
    
    # Generate title if not specified
    title = args.title
    if not title:
        metric_display = args.metric_name.replace('_', ' ').title()
        title = f"Comparison: {metric_display}"
    
    # Generate default output filename if not specified
    output = args.output
    if not output:
        output = f"comparison_{args.metric_name.replace('.csv', '')}.png"
    
    # Plot
    plot_scatter(
        csv_files=csv_files,
        output=output,
        title=title,
        ylabel=args.ylabel,
        labels=args.labels,
        style=args.style,
        show_stats=True,
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
