# Dimer Optimization Toolkit

This directory contains scripts for optimizing and maximizing the yield of a dimer system using JAX-based scientific computation.

## Overview

There are two main workflows:

1. **Maximize Dimer Yield**
   - Use `maximize.py` to maximize the yield of the dimer system by optimizing patch-patch interaction strengths and other parameters.
   - No target yield is specified; the script simply finds the parameter set that gives the highest possible yield.

2. **Optimize for a Target Yield**
   - Use `optimize.py` to optimize parameters so that the dimer system achieves a user-specified target yield.
   - The script minimizes the difference between the achieved yield and the target value.

## Usage

### 1. Maximize Dimer Yield

Run the following command, specifying initial values for patchy interaction, temperature, and concentration:

```bash
python maximize.py --init_patchy_vals 4.5 --init_kt 0.5 --init_conc_val 0.0005
```

- The script will print progress and save results in the `Maximized_agnese/` directory.
- Only the patch-patch interaction strengths are optimized by default (see the `mask` in the script).

### 2. Optimize for a Target Yield

Run the following command (see `optimize.py` for details and options):

```bash
python optimize.py [options]
```

- You can specify a target yield and other advanced options in the script.
- The script will print progress and save results in the `Agnese/` directory.

## Notes
- Both scripts use JAX, Optax, and related scientific libraries.
- All output files are saved in their respective results directories.
- You can adjust which parameters are optimized by editing the `mask` array in each script.

## Contact
For questions or issues, please contact the repository maintainer.
