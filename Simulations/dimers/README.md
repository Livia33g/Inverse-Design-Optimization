# Dimer Simulation Workflow (Signac/HOOMD)

This directory contains scripts and configuration for running dimer simulations using [Signac](https://signac.io/) and [HOOMD-blue](https://hoomd-blue.readthedocs.io/).

## Workflow Overview

1. **Parameter File:**
   - All simulation parameters (e.g., concentrations, epsilon values, etc.) should be specified in a parameter file (e.g., `all_params.txt`).
   - Each line in this file defines a set of simulation conditions for a dimer case.

2. **Job Initialization:**
   - Run:
     ```bash
     python init.py
     ```
   - This reads the parameter file, computes the appropriate box size and dimer counts for each case, and initializes Signac jobs for each parameter set and for multiple replicas per case.

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

## Environment
- **You must use the `hoomd4GPU` environment** to run these simulations. This environment includes HOOMD-blue with GPU support and all required dependencies.
- A `requirements.txt` or `environment.yml` should be provided to match the `hoomd4GPU` environment.

## Summary
- Edit your parameter file to define your simulation cases.
- Run `init.py` to set up jobs, then use `project.py submit` to launch them.
- Monitor with `project.py status` and resubmit as needed for multi-stage workflows.
- All results and job data will be organized in `workspace/` as per the Signac/Flow process.

For more details on customizing the workflow or simulation parameters, see the comments in the scripts or contact the project maintainer.
