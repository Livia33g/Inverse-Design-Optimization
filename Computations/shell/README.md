# Octahedral Shell Assembly Optimization

This directory contains scripts for optimizing the assembly of octahedral shells using rigid-body simulations and partition function calculations with JAX.

## Main Scripts
- `optiomize_octa.py`: **Maximizes or targets a specific yield** for octahedral shell assembly at a single temperature. Use this script to find parameters that optimize yield for a given set of conditions.
- `range.py`: **Finds parameters that minimize yield at a high temperature and maximize yield at a low temperature.** This is useful for designing systems with strong temperature-dependent assembly (e.g., switch-like behavior).
- `sigma_oct.py`: Computes symmetry numbers for each .pos file representing a (possibly incomplete) octahedral structure. This is used for partition function corrections and is only needed if you want to understand or modify the symmetry number approximations for incomplete shells.

## Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare input files:**
   Place the following files in the `octahedron/` directory:
   - `rb_orientation_vec.npy`
   - `rb_center.npy`
   - `vertex_shape_points.npy`
   - `vertex_shape_point_species.npy`

3. **Run the optimization (single temperature, maximize/target yield):**
   ```bash
   python optiomize_octa.py --kt 5.0 --init_morse 12.0 --rep_alpha 1.0 --init_conc 0.001 --number_mon 343 --desired_yield 1.0
   ```
   All arguments are optional and have sensible defaults. Use `-h` for help.

4. **Run the range optimization (minimize yield at high T, maximize at low T):**
   ```bash
   python range.py --kt_low 0.2 --kt_high 1.5 --init_morse 5.0 --rep_alpha 1.0 --init_conc 0.001 --number_mon 600 --desired_yield_h 0.05 --desired_yield_l 1.0
   ```
   This script will search for parameters that suppress assembly at high temperature and promote it at low temperature. All arguments are optional and have sensible defaults. Use `-h` for help.

5. **(Optional) Compute symmetry numbers for incomplete octahedral structures:**
   - Place your `.pos` files (one for each structure) in a folder, e.g. `oct_files/`.
   - Run:
     ```bash
     python sigma_oct.py
     ```
   - This will output a file `symmetry_numbers_oct.txt` with the symmetry number for each structure. This is used for partition function corrections in the main optimization scripts, especially for incomplete shells.

## Arguments for `optiomize_octa.py`
- `--kt`: Initial temperature (default: 5.0)
- `--init_morse`: Initial Morse epsilon (default: 12.0)
- `--rep_alpha`: Initial repulsion alpha (default: 1.0)
- `--init_conc`: Initial concentration (default: 0.001)
- `--number_mon`: Number of monomers (default: 343)
- `--desired_yield`: Desired yield (default: 1.0)

## Arguments for `range.py`
- `--kt_low`: Low temperature (default: 0.2)
- `--kt_high`: High temperature (default: 1.5)
- `--init_morse`: Initial Morse epsilon (default: 5.0)
- `--rep_alpha`: Initial repulsion alpha (default: 1.0)
- `--init_conc`: Initial concentration (default: 0.001)
- `--number_mon`: Number of monomers (default: 600)
- `--desired_yield_h`: Desired yield at high temperature (default: 0.05)
- `--desired_yield_l`: Desired yield at low temperature (default: 1.0)

## Output
- Results and optimized parameters will be saved in the `optimized_results/` or `optimized_range_results/` directory, depending on the script.

## Notes
- Ensure all dependencies are installed and input files are present.
- The scripts are PEP8-compliant, robust, and documented for publication.
- The symmetry number script (`sigma_oct.py`) is only needed if you want to recalculate or inspect symmetry corrections for incomplete shells.
- For questions or issues, please contact the project maintainer.
