# Replica Simulation Workflow: Low and High Temperature Runs by Percentage

This directory contains scripts and instructions for running self-assembly simulations at both low and high temperatures for each percentage increase value. The workflow is designed to organize results for easy analysis and reproducibility.

## Running Simulations

You now use a single SLURM script (`replica.sh`) and a single simulation script (`simulate.py`) for both optimization and range workflows. The output directory is controlled by the `--mode` argument (`opt` or `range`).

### 1. Set Up and Submit Jobs
- For each replica and mode, submit the job as follows:
  ```bash
  sbatch --export=ALL,REPLICA=<replica_id>,MODE=<opt|range> replica.sh
  ```
  - Replace `<replica_id>` with the replica number (e.g., 1, 2, 3).
  - Set `MODE=opt` for optimization output (saved in `optimized/opt_<kT_init>/<percent>`).
  - Set `MODE=range` for range/percentage output (saved in `percentages/percent_<kT_init>/<percent>`).

### 2. Output Organization
- For `MODE=opt`, results are saved in:
  ```
  optimized/opt_<kT_init>/<percent>/
  ```
- For `MODE=range`, results are saved in:
  ```
  percentages/percent_<kT_init>/<percent>/
  ```
- Each directory contains output files for each replica, with filenames indicating the job, percent, temperature, and replica number.

### 3. Parameter File
- The SLURM script reads parameters from the file specified by `PARAM_FILE` (e.g., `opt15.txt`).
- Make sure this file is set correctly in `replica.sh` for your workflow.

### 4. Example Submission
To run 3 replicas for optimization:
```bash
for r in 1 2 3; do
  sbatch --export=ALL,REPLICA=$r,MODE=opt replica.sh
  sbatch --export=ALL,REPLICA=$r,MODE=range replica.sh
done
```

### 5. Notes
- The same `simulate.py` script is used for both workflows; the output directory is determined by the `--mode` argument.
- The SLURM job name, temperature, and other parameters are set automatically from the parameter file and environment variables.

For further details, see comments in the scripts or contact the project maintainer.
