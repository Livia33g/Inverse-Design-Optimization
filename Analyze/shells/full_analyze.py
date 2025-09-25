#!/usr/bin/env python3
import numpy as np
import freud
import os
import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# ==============================================================================
# STEP 1: .pos FILE PARSER
# ==============================================================================
def reconstruct_centers_from_pos(pos_file_path):
    print(f"--> Reconstructing vertex positions from {pos_file_path}...")
    with open(pos_file_path, 'r') as f: lines = f.readlines()
    all_frames_lines = []; current_frame_lines = []
    for line in lines:
        if 'eof' in line: all_frames_lines.append(current_frame_lines); current_frame_lines = []
        else: current_frame_lines.append(line.strip())
    all_centers = []; box_size = None
    for i, frame_lines in enumerate(all_frames_lines):
        vertex_positions = []
        for line in frame_lines:
            parts = line.split()
            if not parts: continue
            if parts[0] == 'boxMatrix':
                if box_size is None: box_size = float(parts[1])
                continue
            if parts[0] == 'def': continue
            if parts[0] == 'V':
                try: pos = [float(p) for p in parts[1:]]; vertex_positions.append(pos)
                except (ValueError, IndexError): pass
        if not vertex_positions:
            if i > 0: print(f"Warning: No 'V' particles found in frame {i}.")
            continue
        all_centers.append(np.array(vertex_positions))
    if not all_centers: print("Error: No valid frames were parsed."); return None, None
    print(f"--> Reconstruction successful. Found {len(all_centers)} frames and {all_centers[0].shape[0]} vertices per frame.")
    return all_centers, box_size

# ==============================================================================
# STEP 2: ANALYSIS WITH CORRECTED COUNTING
# ==============================================================================
def run_full_analysis(reconstructed_centers, box_size, output_dir):
    print("\n--> Starting analysis pipeline...")
    BOND_CUTOFF = 4.2
    TARGET_SIZE = 6
    TARGET_BONDS = 4
    SENSITIVITY_THRESHOLD = 0.1
    
    TOTAL_PARTICLES = reconstructed_centers[0].shape[0]
    box = freud.box.Box.from_box([box_size, box_size, box_size, 0, 0, 0])
    
    num_frames_to_analyze = min(10, len(reconstructed_centers))
    frames_to_analyze = reconstructed_centers[-num_frames_to_analyze:]
    
    total_shells_found = 0
    total_monomers_found = 0
    
    print("\n--- Running Analysis with Corrected Counting Logic ---")
    print(f"    Sensitivity set to: {SENSITIVITY_THRESHOLD:.0%}")
    for i, positions in enumerate(frames_to_analyze):
        cl = freud.cluster.Cluster()
        cl.compute((box, positions), neighbors={"mode": "ball", "r_max": BOND_CUTOFF})
        
        current_frame_cluster_sizes = np.array([len(key) for key in cl.cluster_keys])
        frame_monomers = np.sum(current_frame_cluster_sizes == 1)
        total_monomers_found += frame_monomers
        
        frame_shells = 0
        for cluster_id, cluster_size in enumerate(current_frame_cluster_sizes):
            if cluster_size < TARGET_SIZE:
                continue
    
            cluster_indices = cl.cluster_keys[cluster_id]
            original_cluster_positions = positions[cluster_indices]
            wrapped_cluster_positions = box.wrap(original_cluster_positions)
            
            aq = freud.AABBQuery(box, wrapped_cluster_positions)
            nlist = aq.query(wrapped_cluster_positions, {'mode': 'ball', 'r_max': BOND_CUTOFF}).toNeighborList()
            internal_bond_counts = np.bincount(nlist.query_point_indices, minlength=cluster_size)
    
            num_shell_like_particles = np.sum(internal_bond_counts >= TARGET_BONDS)
            fraction_shell_like = num_shell_like_particles / cluster_size
            
            if fraction_shell_like >= SENSITIVITY_THRESHOLD:
                shells_in_this_cluster = cluster_size // TARGET_SIZE
                frame_shells += shells_in_this_cluster
    
        total_shells_found += frame_shells
        print(f"    Frame {i+1}/{num_frames_to_analyze}: Found {frame_shells} shells and {frame_monomers} monomers.")
    
    # --- Finalize and Save Results ---
    avg_shells = total_shells_found / num_frames_to_analyze
    avg_monomers = total_monomers_found / num_frames_to_analyze
    yield_as_particle_fraction = (avg_shells * TARGET_SIZE) / TOTAL_PARTICLES
    monomer_fraction = avg_monomers / TOTAL_PARTICLES
    accounted_particle_fraction = yield_as_particle_fraction + monomer_fraction
    
    yield_results = {
        "analysis_method": "Generous Internal Connectivity",
        "parameters": { "sensitivity_threshold": SENSITIVITY_THRESHOLD, "total_particles": TOTAL_PARTICLES },
        "results": {
            "average_shells_per_frame": avg_shells,
            "average_monomers_per_frame": avg_monomers,
            "yield_as_particle_fraction": yield_as_particle_fraction,
            "monomer_fraction": monomer_fraction,
            "accounted_particle_fraction (shells + monomers)": accounted_particle_fraction
        }
    }
    json_path = os.path.join(output_dir, "shell_yield.json")
    with open(json_path, 'w') as f: json.dump(yield_results, f, indent=4)
    
    print("\n--- Yield Summary ---")
    print(f"    Average shells formed per frame: {avg_shells:.2f}")
    print(f"    Yield (fraction of vertices in shells): {yield_as_particle_fraction:.2%}")
    print("------------------------------------------")
    print(f"    Saved detailed yield data to {json_path}")


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Analyze a .pos file for shell formation yield.")
    parser.add_argument("pos_file", type=str, help="Path to the input .pos trajectory file.")
    args = parser.parse_args()
    
    pos_input = Path(args.pos_file)
    if not pos_input.is_file():
        print(f"Error: Input file not found at '{pos_input}'")
        return

    # ensure top-level workspace dir exists
    workspace_dir = Path("workspace")
    workspace_dir.mkdir(exist_ok=True)
    
    # create per-file analysis folder under workspace
    output_dir = workspace_dir / f"{pos_input.stem}_analysis"
    output_dir.mkdir(exist_ok=True)
    
    reconstructed_centers, box_size = reconstruct_centers_from_pos(str(pos_input))
    if reconstructed_centers is None:
        print("Analysis stopped.")
        return
    
    run_full_analysis(reconstructed_centers, box_size, str(output_dir))
    print(f"\n✅ Analysis complete for {pos_input.name}!")

if __name__ == "__main__":
    main()