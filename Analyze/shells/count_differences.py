#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import defaultdict
import numpy as np
import json

def get_yield_from_json(filename, ws_dir):
    """Finds the corresponding json file in the workspace and returns its yield."""
    json_path = ws_dir / f"{filename.stem}_analysis" / "shell_yield.json"
    if not json_path.is_file(): return None
    with open(json_path, 'r') as f: data = json.load(f)
    return data.get("results", {}).get("yield_as_particle_fraction")

def generate_plot_for_directory(root_data_dir, workspace_dir, x_axis_label, output_base_name, master_style_map):
    """
    A generic function that processes a data directory and generates a publication-quality plot.
    """
    print(f"\n{'='*20} Processing: {root_data_dir} {'='*20}")
    data_dir = Path(root_data_dir)
    if not data_dir.is_dir():
        print(f"❌ Warning: Directory '{root_data_dir}' not found. Skipping.")
        return

    # Find all the deepest subdirectories that contain the actual .pos files.
    leaf_dirs = sorted(list(set(p.parent for p in data_dir.rglob("low_*.pos"))))
    if not leaf_dirs:
        print(f"❌ Warning: No subdirectories with 'low_*.pos' files found in '{root_data_dir}'. Skipping.")
        return

    all_differences = []
    for leaf_dir in leaf_dirs:
        low_files = list(leaf_dir.glob("low_*.pos"))
        other_files = [f for f in list(leaf_dir.glob("*.pos")) if f not in low_files]
        if not low_files or not other_files: continue
        
        low_yields = [y * 100 for y in [get_yield_from_json(f, workspace_dir) for f in low_files] if y is not None]
        other_yields = [y * 100 for y in [get_yield_from_json(f, workspace_dir) for f in other_files] if y is not None]
        if not low_yields or not other_yields: continue

        avg_low, std_low = np.mean(low_yields), np.std(low_yields)
        avg_other, std_other = np.mean(other_yields), np.std(other_yields)
        yield_diff = abs(avg_other - avg_low)
        error_diff = np.sqrt(std_low**2 + std_other**2)

        # --- THIS IS THE CORRECTED PARAMETER EXTRACTION LOGIC ---
        # The line color is always defined by the kT value from a filename.
        kt_match = re.search(r'kT_([\d\.]+)', other_files[0].name)
        kt_val = float(kt_match.group(1)) if kt_match else None
        
        # The x-axis value is ALWAYS the name of the directory containing the 6 files.
        # This regex just finds any number in the directory name.
        x_val_match = re.search(r'(\d+\.?\d*)', leaf_dir.name)
        x_val = float(x_val_match.group(1)) if x_val_match else None
        
        if x_val is not None and kt_val is not None:
            all_differences.append({
                'x_value': x_val, 'kt_value': kt_val,
                'yield_difference': yield_diff, 'propagated_error': error_diff,
            })
    
    if not all_differences:
        print(f"❌ Warning: After processing, no valid data points were found for '{root_data_dir}'.")
        return

    df = pd.DataFrame(all_differences)
    print(f"✅ Successfully processed {len(df)} difference points for '{root_data_dir}'.")

    # --- PLOTTING LOGIC IS GENERIC AND UNCHANGED ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    grouped_data = df.groupby('kt_value')
    
    for kt_val, group in grouped_data:
        group = group.sort_values('x_value')
        style = master_style_map.get(kt_val)
        if not style: continue
        
        ax.errorbar(
            x=group['x_value'], y=group['yield_difference'], yerr=group['propagated_error'],
            label=f"kT = {kt_val}", **style, linewidth=2.5, markersize=9, capsize=5,
            elinewidth=1.5, capthick=1.5
        )

    ax.set_title(f'Yield Difference vs. {x_axis_label}', fontsize=20, fontweight='bold', pad=15)
    ax.set_xlabel(x_axis_label, fontsize=16)
    ax.set_ylabel('Absolute Yield Difference (%)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.2)
    ax.legend(title='Temperature', fontsize=12, title_fontsize=13, loc='best')
    ax.set_ylim(bottom=0)
    for spine in ax.spines.values(): spine.set_linewidth(1.2)
    plt.tight_layout()

    png_path = f"{output_base_name}.png"
    svg_path = f"{output_base_name}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved plot to: {png_path} and {svg_path}")

def main():
    workspace_dir = Path("workspace")
    if not workspace_dir.is_dir():
        print("❌ Error: 'workspace' directory not found. Please run the main analysis jobs first.")
        return

    # --- Create a master style guide for consistent plotting ---
    print("--- Creating a consistent style map for all plots ---")
    all_kts = set()
    for root_dir in ["percentages", "additions"]:
        if Path(root_dir).is_dir():
            for f in Path(root_dir).rglob("*.pos"):
                kt_match = re.search(r'kT_([\d\.]+)', f.name)
                if kt_match: all_kts.add(float(kt_match.group(1)))
    
    unique_kts = sorted(list(all_kts))
    colors, markers, linestyles = plt.get_cmap('tab10').colors, ['o', 's', '^', 'D', 'v'], ['-', '--', '-.', ':']
    MASTER_STYLE_MAP = {kt: {'color': colors[i%len(colors)], 'marker': markers[i%len(markers)], 'linestyle': linestyles[i%len(linestyles)]} for i, kt in enumerate(unique_kts)}
    print(f"✅ Style map created for {len(unique_kts)} unique temperatures.")
    
    # --- Run for the 'percentages' directory ---
    generate_plot_for_directory(
        root_data_dir="percentages", workspace_dir=workspace_dir,
        x_axis_label="Percent Increase", output_base_name="percent_increase_plot",
        master_style_map=MASTER_STYLE_MAP
    )

    # --- Run for the 'additions' directory ---
    generate_plot_for_directory(
        root_data_dir="additions", workspace_dir=workspace_dir,
        x_axis_label="Linear Increase", output_base_name="linear_increase_plot",
        master_style_map=MASTER_STYLE_MAP
    )
    print("\n\n🎉 All plotting tasks complete!")

if __name__ == "__main__":
    main()