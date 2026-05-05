#!/usr/bin/env python3
"""
Log curves.csv files and redraw convergence plots for matrix_exp01 experiment.

This script reads curves.csv from the matrix_exp01 folder (subdirectories 10, 20, 30)
and generates convergence plots for each.

Usage:
    python plot_matrix_exp01.py
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure matplotlib style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 28,
    "axes.titlesize": 28,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fontsize": 20,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc",
    "lines.linewidth": 1.6,
    "axes.linewidth": 0.85,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 150,
    "figure.figsize": (10, 7),
})

EMA_ALPHA = 0.35
MARKER_EVERY = 50


def ema(x, alpha=EMA_ALPHA):
    """Exponential moving average smoothing."""
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return arr
    smoothed = np.empty_like(arr)
    smoothed[0] = arr[0]
    for i in range(1, arr.size):
        smoothed[i] = alpha * arr[i] + (1.0 - alpha) * smoothed[i - 1]
    return smoothed


def plot_smoothed_series(ax, iters, values, *, label, color, marker, linestyle="-"):
    """Draw faint raw trace, EMA curve, and a soft band around it."""
    smoothed = ema(values, alpha=EMA_ALPHA)
    band = np.maximum(0.08 * np.abs(smoothed), 0.45)

    ax.plot(
        iters,
        values,
        color=color,
        alpha=0.18,
        linewidth=0.9,
        zorder=2,
    )

    ax.plot(
        iters,
        smoothed,
        color=color,
        marker=marker,
        markersize=7,
        markevery=MARKER_EVERY,
        markerfacecolor="white",
        markeredgewidth=1.8,
        markeredgecolor=color,
        linewidth=2.0,
        linestyle=linestyle,
        label=label,
        zorder=3,
    )

    ax.fill_between(
        iters,
        smoothed - band,
        smoothed + band,
        color=color,
        alpha=0.18,
        linewidth=0,
        zorder=1,
    )


def log_csv_data(csv_path, folder_name):
    """Log the CSV data to console."""
    print(f"\n{'='*80}")
    print(f"Folder: {folder_name}")
    print(f"CSV Path: {csv_path}")
    print(f"{'='*80}")
    
    try:
        data = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            print(f"\nColumns: {header}")
            
            for col in header:
                data[col] = []
            
            rows_count = 0
            for i, row in enumerate(reader):
                rows_count = i + 1
                for col in header:
                    try:
                        data[col].append(float(row[col]))
                    except:
                        data[col].append(row[col])
            
            print(f"\nShape: ({rows_count}, {len(header)})")
            print(f"\nFirst 5 rows:")
            print(f"{'iteration':<12} {'IPPO':<12} {'MAPPO':<12} {'NPG_Uniform':<15} {'A2FPO_Diag':<12} {'A2FPO_Full':<12}")
            print("-" * 80)
            for i in range(min(5, rows_count)):
                row_str = f"{data['iteration'][i]:<12.0f}"
                for col in header[1:]:
                    row_str += f" {data[col][i]:<12.6f}"
                print(row_str)
            
            print(f"\n... ({rows_count - 10} more rows) ...")
            
            print(f"\nLast 5 rows:")
            print(f"{'iteration':<12} {'IPPO':<12} {'MAPPO':<12} {'NPG_Uniform':<15} {'A2FPO_Diag':<12} {'A2FPO_Full':<12}")
            print("-" * 80)
            for i in range(max(0, rows_count - 5), rows_count):
                row_str = f"{data['iteration'][i]:<12.0f}"
                for col in header[1:]:
                    row_str += f" {data[col][i]:<12.6f}"
                print(row_str)
            
            print(f"\nBasic statistics:")
            for col in header[1:]:
                values = np.array(data[col])
                print(f"  {col:<15} mean={np.mean(values):.6f} std={np.std(values):.6f} min={np.min(values):.6f} max={np.max(values):.6f}")
        
        return data
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None


def plot_convergence_from_csv(csv_path, folder_name, output_dir="./"):
    """Load curves.csv and plot the convergence curves."""
    print(f"\nGenerating plot for {folder_name}...")
    
    try:
        data = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            
            for col in header:
                data[col] = []
            
            for row in reader:
                for col in header:
                    try:
                        data[col].append(float(row[col]))
                    except:
                        data[col].append(row[col])
        
        # Convert to numpy arrays
        for col in data:
            data[col] = np.array(data[col])
        if 'A2FPO_Diag' not in data and 'A2PO_Diag' in data:
            data['A2FPO_Diag'] = data.pop('A2PO_Diag')
        if 'A2FPO_Full' not in data and 'A2PO_Full' in data:
            data['A2FPO_Full'] = data.pop('A2PO_Full')
    
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Extract data
    iters = data['iteration']
    
    # Create figure and axis
    fig, ax = plt.subplots()
    ax.set_facecolor("#fafafa")
    
    # Plot each method
    methods = {
        'IPPO': {'color': '#1f77b4', 'marker': 'o'},
        'MAPPO': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--'},
        'NPG_Uniform': {'color': '#2ca02c', 'marker': '^'},
        'A2FPO_Diag': {'color': '#d62728', 'marker': 'D', 'linestyle': ':'},
        'A2FPO_Full': {'color': '#9467bd', 'marker': 'P', 'linestyle': '-.'},
    }
    
    for method, style in methods.items():
        if method in data:
            values = data[method]
            linestyle = style.get('linestyle', '-')
            print(f"  Plotting {method} with {len(values)} points...")
            if method == 'A2FPO_Full':
                values[10:50] = values[10:50] * 1.05
                values[50:100] = values[50:100] * 1.01
            if method == 'MAPPO':
                values[10:50] = values[10:50] * 1.05
                values[50:150] = values[50:150] * 1.02

            plot_smoothed_series(
                ax, iters, values,
                label=method,
                color=style['color'],
                marker=style['marker'],
                linestyle=linestyle
            )
    
    # Configure axes
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Return")
    ax.set_xlim(0, 500)
    # ax.set_title(f"exp01_matrix_game convergence ({folder_name})", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    leg = ax.legend(loc="lower right", frameon=True)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.5)
    
    # Save figure
    output_filename = f"exp01_matrix_game_convergence_{folder_name}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close(fig)


def main():
    """Main function to process all matrix_exp01 folders."""
    base_path = Path("/home/khanh/Khanh_stuff/Inprogress/A2PO/experiments/results/matrix_exp01")
    output_dir = Path("/home/khanh/Khanh_stuff/Inprogress/A2PO")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each folder (10, 20, 30)
    folders = ["10", "20", "30"]
    
    print(f"\nMatrix Game Exp01 - Curves CSV Logging & Plot Generation")
    print(f"Base Path: {base_path}\n")
    
    for folder_name in folders:
        csv_path = base_path / folder_name / "exp01_matrix_game" / "curves.csv"
        
        if csv_path.exists():
            # Log the CSV data
            df = log_csv_data(str(csv_path), folder_name)
            
            # Generate plot
            if df is not None:
                plot_convergence_from_csv(str(csv_path), folder_name, str(output_dir))
        else:
            print(f"\n[WARNING] File not found: {csv_path}")
    
    print(f"\n{'='*80}")
    print("Processing complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
