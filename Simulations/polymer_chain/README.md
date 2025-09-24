# Polymer Chain Simulation Workflow (Signac/HOOMD)

This directory contains scripts and configuration for running polymer chain simulations using [Signac](https://signac.io/) and [HOOMD-blue](https://hoomd-blue.readthedocs.io/).

## Workflow Overview

1. **Parameter File:**
   - All simulation parameters (concentrations, epsilon values, etc.) are specified in `all_params.txt`.
   - Each line in this file defines a set of simulation conditions.

2. **Job Initialization:**
   - Run:
     ```bash
     python init.py
     ```
   - This reads `all_params.txt`, computes the appropriate box size and monomer counts for each case, and initializes Signac jobs for each parameter set and for 3 replicas per case.

3. **Project Submission:**
   - Submit jobs to the workflow engine with:
     ```bash
     python project.py submit
     ```
   - This will launch the initialization and equilibration operations for all jobs, as defined in the Signac FlowProject.
   - All jobs and their data will appear in the `workspace/` directory, managed by Signac.

4. **Monitoring Job Status:**
   - Check the status of all jobs with:
     ```bash
     python project.py status
     ```
   - Once the `initialize` operation has completed and the `equilibrate` operation is available, you can resubmit jobs for equilibration:
     ```bash
     python project.py submit
     ```

## About Signac/Flow
- [Signac](https://signac.io/) is a data and workflow management tool for computational research.
- [Signac Flow](https://signac-flow.readthedocs.io/) extends Signac to manage complex workflows and job dependencies.
- All job data and results are stored in the `workspace/` directory, and job status is tracked automatically.

## Environment Setup

Before running any scripts, set up the required Python environment:

- If using pip:
  ```bash
  python -m venv hoomd4GPU_env
  source hoomd4GPU_env/bin/activate
  pip install -r requirements_hoomd4GPU.txt
  ```

- If using conda (if you have an `environment.yml`):
  ```bash
  conda env create -f environment.yml
  conda activate hoomd4GPU
  ```

This ensures all dependencies (including HOOMD-blue with GPU support) match the tested environment.

## Summary
- Edit `all_params.txt` to define your simulation cases.
- Run `init.py` to set up jobs, then use `project.py submit` to launch them.
- Monitor with `project.py status` and resubmit as needed for multi-stage workflows.
- All results and job data will be organized in `workspace/` as per the Signac/Flow process.

For more details on customizing the workflow or simulation parameters, see the comments in the scripts or contact the project maintainer.
