# Polymer Chain Assembly Optimization Toolkit

This repository provides a robust and reproducible workflow for optimizing interaction parameters and concentrations in polymer chain assembly, with a focus on maximizing the yield of a target structure under physical and mass action constraints.

## Workflow Overview

The optimization workflow requires two key input data files:
- `species_all_upto3.pkl`: Contains all species combinations for up to 3 monomer species and up to 3-mers.
- `species_targetsize4.pkl`: Contains all species combinations for 4-mers (tetramers) for 3 monomer species.

These files are generated using the provided scripts and should be placed in an `input_data/` directory for clarity and reproducibility.

### Step 1: Generate Combinations with `all_comb_sigma.py`

First, generate all possible species combinations for 3 monomer species and up to 4-mers:

```bash
python all_comb_sigma.py --num_monomers 3 --max_size 4 --output_dir input_data
```

This will create a file (e.g., `input_data/species_combinations_3_4.pkl`) containing all combinations up to 4-mers.

### Step 2: Extract Tetramer Data with `extract_4.py`

Next, use the file from Step 1 to extract and save only the tetramer (4-mer) data:

```bash
python extract_4.py --input input_data/species_combinations_3_4.pkl --output input_data/species_targetsize4.pkl
```

This produces `input_data/species_targetsize4.pkl` for use in optimization.

### Step 3: Generate 3-mer Data for General Use

Now, re-run `all_comb_sigma.py` to generate all species combinations for 3 monomer species and up to 3-mers:

```bash
python all_comb_sigma.py --num_monomers 3 --max_size 3 --output_dir input_data
```

This will create `input_data/species_all_upto3.pkl`.

### Step 4: Run the Optimization

With both `species_all_upto3.pkl` and `species_targetsize4.pkl` in the `input_data/` directory, you can now run the main optimization script:

```bash
python optimize.py --input_dir input_data --output results.txt [other options]
```

See the script's `--help` flag for all available options.

## Advanced Options and Customization

### Disabling the Mass Action Constraint
By default, the mass action constraint is enabled if you use the `--use_mass_action` flag. If you want to disable the mass action constraint in the code, you can comment/uncomment the relevant line in the loss function (look for the `#FIXME` comment):

```python
# In optimize.py or Mass_all.py, inside optimize_grad_fn:
# mass_act_loss = 1 #FIXME uncomment if not using mass action constraint
```

Alternatively, use the `--use_mass_action` flag to toggle this behavior at runtime.

### Selective Parameter Optimization (Masking Gradients)
The code uses a `masked_grads` function to allow selective optimization of parameters. By editing the `mask` array in the script, you can fix (not optimize) specific parameters such as temperature, concentrations, or interaction strengths. For example, to fix temperature during optimization, set the corresponding entry in the mask to `0.0`:

```python
# Example: To fix temperature (kT), set its mask entry to 0.0
mask = mask.at[kT_index].set(0.0)
```

This allows you to:
- Fix temperature or other parameters during optimization
- Only optimize a subset of parameters (e.g., only concentrations or only interaction strengths)

### Custom Pairwise Interaction Optimization
The code uses a `custom_pairs` list to specify which pairwise interaction strengths (epsilon values) are optimized during the run. This allows you to:
- Focus optimization on a subset of all possible monomer-monomer or patch-patch interactions.
- Easily add or remove pairs from the optimization by editing the `custom_pairs` list in the script.

For example, in `optimize.py` or `Mass_all.py`:

```python
custom_pairs = [
    (2, 3), (4, 5), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6), (4, 6), (5, 6),
    (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
]
```

- Only the interaction strengths for these pairs will be optimized.
- If you want to optimize all possible pairs, you can generate all combinations and assign to `custom_pairs`.
- If you want to fix certain interactions, simply remove them from the list or set their mask entry to `0.0` in the `mask` array.

This provides fine-grained control over which physical interactions are tuned during optimization, supporting both targeted and global optimization scenarios.

## Notes
- All scripts support user-friendly command-line arguments for reproducibility.
- The `input_data/` directory is recommended for clarity, but you may use another directory if you update the script arguments accordingly.
- For best reproducibility, document your environment (see `requirements.txt` or `environment.yml`).

## Example Full Workflow
```bash
# Step 1: Generate all combinations up to 4-mers
python all_comb_sigma.py --num_monomers 3 --max_size 4 --output_dir input_data

# Step 2: Extract tetramer data
python extract_4.py --input input_data/species_combinations_3_4.pkl --output input_data/species_targetsize4.pkl

# Step 3: Generate all combinations up to 3-mers
python all_comb_sigma.py --num_monomers 3 --max_size 3 --output_dir input_data

# Step 4: Run optimization
python optimize.py --input_dir input_data --output results.txt --use_mass_action
```

## Contact
For questions or issues, please contact the repository maintainer.
