#!/usr/bin/env python3
import json
import csv
from pathlib import Path
import re  # We need the regular expression module to find the '_repX' part
import numpy as np # Numpy is perfect for calculating mean and standard deviation

def main():
    """
    Finds all 'shell_yield.json' files, groups them by a base name
    (ignoring _repX), calculates the average and standard deviation for each
    group, and writes a final summary CSV file.
    """
    workspace_dir = Path("workspace")
    if not workspace_dir.is_dir():
        print(f"❌ Error: 'workspace' directory not found.")
        print("Please ensure the SLURM jobs have completed successfully.")
        return

    json_files = sorted(list(workspace_dir.glob("*_analysis/shell_yield.json")))
    
    if not json_files:
        print(f"❌ Error: No 'shell_yield.json' files found in '{workspace_dir}'.")
        return

    print(f"✅ Found {len(json_files)} individual result files. Grouping and averaging...")

    # Step 1: Group the yields by a common base name
    # The dictionary will look like:
    # { "basename1": [yield1, yield2, yield3], "basename2": [yieldA, yieldB] }
    grouped_yields = {}

    for json_path in json_files:
        # Get the original filename, e.g., "my_sim_kT_1.1_rep1.pos"
        original_filename = json_path.parent.name.replace("_analysis", ".pos")

        # Create the 'base_name' by removing the replicate identifier
        # This regex looks for "_rep" followed by one or more digits (_rep\d+)
        # and replaces it with an empty string.
        base_name = re.sub(r'_rep\d+\.pos$', '.pos', original_filename)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        yield_fraction = data.get("results", {}).get("yield_as_particle_fraction")
        
        if yield_fraction is not None:
            # If the base_name isn't in our dictionary yet, add it with an empty list
            if base_name not in grouped_yields:
                grouped_yields[base_name] = []
            
            # Append the current yield (as a percentage) to the list for that group
            grouped_yields[base_name].append(yield_fraction * 100)

    # Step 2: Calculate the average and standard deviation for each group
    averaged_results = []
    for base_name, yield_list in grouped_yields.items():
        avg_yield = np.mean(yield_list)
        std_dev_yield = np.std(yield_list)
        num_replicates = len(yield_list)
        
        averaged_results.append({
            'base_filename': base_name,
            'avg_yield_percentage': avg_yield,
            'std_dev_yield': std_dev_yield,
            'num_replicates': num_replicates
        })
        
    # Step 3: Write the final averaged results to a new CSV file
    output_csv_path = Path("summary_averaged_results.csv")
    fieldnames = [
        "base_filename", 
        "avg_yield_percentage", 
        "std_dev_yield",
        "num_replicates"
    ]
    
    # Sort results alphabetically by base filename for a clean output
    averaged_results.sort(key=lambda x: x['base_filename'])

    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(averaged_results)
    
    print("-" * 50)
    print(f"✅ Successfully created averaged summary file: {output_csv_path}")
    print(f"   Processed {len(json_files)} files into {len(averaged_results)} averaged groups.")
    print("-" * 50)


if __name__ == "__main__":
    main()