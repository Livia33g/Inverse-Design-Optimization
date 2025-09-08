#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import re
import numpy as np
import json
import sys


# This helper function is unchanged and essential.
def get_yield_from_json(filename, ws_dir):
    """Finds the corresponding json file in the workspace and returns its yield."""
    json_path = ws_dir / f"{filename.stem}_analysis" / "shell_yield.json"
    if not json_path.is_file():
        return None
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data.get("results", {}).get("yield_as_particle_fraction")
    except (json.JSONDecodeError, KeyError):
        print(f"   -> Warning: Could not read or parse {json_path}")
        return None


# This data processing function is also unchanged. It's robust and gives us the
# per-temperature data we need to perform the new aggregation.
def process_data_source(root_dir_str, file_glob_pattern, data_label, workspace_dir):
    """
    Processes a directory to find files, calculate average yields from replicas,
    and extract parameters (x-value, kT). Returns a list of individual data points.
    """
    print(f"\n{'='*20} Processing Source: {data_label} from '{root_dir_str}' {'='*20}")
    root_dir = Path(root_dir_str)
    if not root_dir.is_dir():
        print(f"❌ Warning: Directory '{root_dir_str}' not found. Skipping.")
        return []

    leaf_dirs = sorted(list(set(p.parent for p in root_dir.rglob(file_glob_pattern))))
    if not leaf_dirs:
        print(
            f"❌ Warning: No subdirectories with '{file_glob_pattern}' files found in '{root_dir_str}'. Skipping."
        )
        return []

    all_points = []
    for leaf_dir in leaf_dirs:
        files_to_process = list(leaf_dir.glob(file_glob_pattern))
        if not files_to_process:
            continue

        yields = [
            y * 100
            for y in [get_yield_from_json(f, workspace_dir) for f in files_to_process]
            if y is not None
        ]
        if not yields:
            continue

        avg_yield = np.mean(yields)
        std_yield = np.std(yields)
        x_val_match = re.search(r"(\d+\.?\d*)", leaf_dir.name)
        x_val = float(x_val_match.group(1)) if x_val_match else None
        kt_match = re.search(r"kT_([\d\.]+)", files_to_process[0].name)
        kt_val = float(kt_match.group(1)) if kt_match else None

        if x_val is not None and kt_val is not None:
            all_points.append(
                {
                    "x_value": x_val,
                    "kt_value": kt_val,
                    "mean_yield": avg_yield,
                    "replica_error": std_yield,
                    "label": data_label,
                }
            )

    if all_points:
        print(
            f"✅ Successfully processed {len(all_points)} data points for '{data_label}'."
        )
    return all_points


def generate_aggregate_plot(all_data_points, kt_marker_map, output_base_name):
    """
    Generates a plot with one aggregated line per data source.
    - The line connects the mean yield across all temperatures for each x-value.
    - A vertical line shows the range (min to max) of yields from different temperatures.
    - Individual temperature points are shown with unique markers on the vertical line.
    """
    if not all_data_points:
        print("❌ Error: No data points were provided for plotting. Aborting.")
        return

    df = pd.DataFrame(all_data_points)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define a style map for the two main data series
    label_style_map = {
        "Theoretical": {"color": "C0", "label": "Theoretical"},  # Blue
        "Optimized": {"color": "C1", "label": "Optimized"},  # Orange
    }

    # --- Main plotting loop: one iteration for "Theoretical", one for "Optimized" ---
    for label, group_df in df.groupby("label"):
        if label not in label_style_map:
            continue

        style = label_style_map[label]

        # 1. CALCULATE THE AGGREGATE LINE
        line_data = (
            group_df.groupby("x_value")["mean_yield"]
            .mean()
            .reset_index()
            .sort_values("x_value")
        )

        # 2. PLOT THE AGGREGATE LINE
        ax.plot(
            line_data["x_value"],
            line_data["mean_yield"],
            color=style["color"],
            label=style["label"],
            linewidth=3,
            marker="o",
            markersize=5,
            zorder=10,
        )

        # 3. PLOT THE CUSTOM RANGE BARS AND TEMPERATURE MARKERS
        for x_val, x_group in group_df.groupby("x_value"):
            min_yield = x_group["mean_yield"].min()
            max_yield = x_group["mean_yield"].max()

            ax.vlines(
                x=x_val,
                ymin=min_yield,
                ymax=max_yield,
                color=style["color"],
                alpha=0.4,
                linewidth=2,
                zorder=1,
            )

            for _, row in x_group.iterrows():
                ax.plot(
                    x_val,
                    row["mean_yield"],
                    marker=kt_marker_map.get(row["kt_value"], "x"),
                    color=style["color"],
                    markersize=9,
                    alpha=0.9,
                    markeredgewidth=1.5,
                    markerfacecolor="white",
                    zorder=5,
                )

    # --- CREATE LEGENDS ---
    first_legend = ax.legend(
        title="Data Source", loc="upper left", fontsize=12, title_fontsize=13
    )
    ax.add_artist(first_legend)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="grey",
            label=f"kT = {kt}",
            linestyle="None",
            markersize=10,
            markerfacecolor="white",
            markeredgewidth=1.5,
        )
        for kt, marker in kt_marker_map.items()
    ]
    ax.legend(
        handles=legend_elements,
        title="Temperature",
        loc="lower right",
        fontsize=12,
        title_fontsize=13,
    )

    # --- Final Plot Styling ---
    ax.set_title(
        "Yield vs. Percent Increase: Theoretical vs. Optimized",
        fontsize=20,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Percent Increase", fontsize=16)
    ax.set_ylabel("Resulting Yield (%)", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    # Save the plot
    png_path = f"{output_base_name}.png"
    svg_path = f"{output_base_name}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✅✅✅ Aggregate trend plot saved to: {png_path} and {svg_path}")


def main():
    workspace_dir = Path("workspace")
    if not workspace_dir.is_dir():
        print(
            f"❌ Error: '{workspace_dir}' directory not found. Please run the analysis jobs first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- DYNAMICALLY CREATE A MASTER MARKER MAP FOR TEMPERATURES ---
    print("--- Creating a consistent marker map for temperatures ---")
    all_kts = set()
    for root_dir in ["percentages", "optimized"]:
        if Path(root_dir).is_dir():
            for f in Path(root_dir).rglob("*.pos"):
                kt_match = re.search(r"kT_([\d\.]+)", f.name)
                if kt_match:
                    all_kts.add(float(kt_match.group(1)))

    unique_kts = sorted(list(all_kts))
    if not unique_kts:
        print(
            "❌ Error: No kT values found in any .pos filenames. Cannot create marker map.",
            file=sys.stderr,
        )
        sys.exit(1)

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    KT_MARKER_MAP = {kt: markers[i % len(markers)] for i, kt in enumerate(unique_kts)}
    print(f"✅ Marker map created for {len(unique_kts)} unique temperatures.")

    # --- Process each data source ---
    # Process the 'high' files from the 'percentages' directory
    theoretical_data = process_data_source(
        root_dir_str="percentages",
        file_glob_pattern="high_*.pos",  # IMPORTANT: Adjust if your filenames differ
        data_label="Optimized",  # <--- NAME CHANGE HERE
        workspace_dir=workspace_dir,
    )

    # Process the replica files from the 'optimized' directory
    optimized_data = process_data_source(
        root_dir_str="optimized",
        file_glob_pattern="opt_*.pos",  # IMPORTANT: Adjust if your filenames differ
        data_label="Theoretical",
        workspace_dir=workspace_dir,
    )

    # --- Combine and Plot ---
    all_data_for_plot = theoretical_data + optimized_data
    generate_aggregate_plot(
        all_data_points=all_data_for_plot,
        kt_marker_map=KT_MARKER_MAP,
        output_base_name="theoretical_vs_optimized_yield_plot",
    )

    print("\n\n🎉 All plotting tasks complete!")


if __name__ == "__main__":
    main()
