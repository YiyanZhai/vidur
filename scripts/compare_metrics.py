#!/usr/bin/env python3
"""
Compare scatter plots from multiple Vidur simulation outputs.

Usage:
    python scripts/compare_metrics.py <metric_name> <output_dir1> <output_dir2> ... [options]
    
Examples:
    # Compare E2E times
    python scripts/compare_metrics.py request_e2e_time_scatter \
        simulator_output/baseline/plots \
        simulator_output/optimized/plots \
        --labels "Baseline" "Optimized" \
        --output comparison_e2e.png
    
    # Compare TTFT
    python scripts/compare_metrics.py request_ttft_scatter \
        simulator_output/config1/plots \
        simulator_output/config2/plots \
        simulator_output/config3/plots \
        --labels "Config1" "Config2" "Config3"
    
    # Compare two metrics from one folder
    python scripts/compare_metrics.py request_e2e_time_scatter prefill_e2e_time_scatter \
        simulator_output/combined_vanilla_1/plots \
        --labels "E2E Time" "TTFT"
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
        'metric_names',
        nargs='+',
        help='Metric CSV filename(s) (without .csv) or with .csv extension'
    )
    parser.add_argument(
        '--output-dirs',
        nargs='+',
        help='Output directories containing plots/ (for multi-folder comparison)'
    )
    parser.add_argument(
        '--labels',
        nargs='+',
        help='Labels for each metric or output directory'
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
    
    # Determine mode: multi-metric (single folder) or multi-folder (single/multiple metrics)
    if args.output_dirs:
        # Multi-folder mode
        metric_names = [args.metric_names[0]]  # Use first metric
        output_dirs = args.output_dirs
    else:
        # Multi-metric mode (last argument is the folder)
        if len(args.metric_names) < 2:
            print("[error] Specify at least one metric and one output directory")
            return 1
        metric_names = args.metric_names[:-1]
        output_dirs = [args.metric_names[-1]]
    
    # Add .csv extension if not present
    metric_names = [m if m.endswith('.csv') else m + '.csv' for m in metric_names]
    
    # Find CSV files
    csv_files = []
    for output_dir in output_dirs:
        for metric_name in metric_names:
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
        if len(metric_names) > 1 and len(output_dirs) == 1:
            title = f"Comparison: Multiple Metrics"
        else:
            metric_display = metric_names[0].replace('.csv', '').replace('_', ' ').title()
            title = f"Comparison: {metric_display}"
    
    # Generate default output filename if not specified
    output = args.output
    if not output:
        if len(metric_names) > 1:
            output = f"comparison_multi_metrics.png"
        else:
            output = f"comparison_{metric_names[0].replace('.csv', '')}.png"
    
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
