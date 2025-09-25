import signac
import numpy as np
import random

project = signac.init_project()

alpha = 5.0
r = 1.0
# replicas = [0, 1, 2]
replicas = [0]
Ns = [3]

# Load kT, target_yield, epsilon from params.txt
params_file = "params.txt"
main_params = []
with open(params_file, 'r') as f:
    for line in f:
        if line.strip():  # skip blank lines
            parts = line.strip().split(',')
            if len(parts) < 5:  # Expecting 5 values now
                continue  # skip incomplete lines
            kT = float(parts[0])
            target_yield = float(parts[1])
            epsilon_1 = float(parts[2])
            epsilon_2 = float(parts[3])
            epsilon_3 = float(parts[4])
            main_params.append((epsilon_1, epsilon_2, epsilon_3, target_yield, kT))

# Loop through parameter combinations and initialize jobs
for N in Ns:
    for replica in replicas:
        for epsilon_1, epsilon_2, epsilon_3, target_yield, kT in main_params:
            sp = {
                'Nx': N,
                'Ny': N,
                'Nz': N,
                'r': r,
                'lat_a': 12.0,
                'seed': random.randint(1, 65535),
                'replica': replica,
                'scale': 0.99,
                'equil_step': 5e4,
                'concentration': 0.001,
                'kT': kT,
                'a': 1.0,
                'b': 0.3,
                'separation': 2.0,
                'alpha': alpha,
                'D0': 1.0,
                'target_yield': target_yield,
                'd_colors': [epsilon_1, epsilon_2, epsilon_3],  # List of epsilons
                'r0': 0.0,
                'r_cut': 8.0 / alpha,
                'rep_A': 500.0,
                'rep_alpha': 2.5,
                'rep_r_min': 0.0,
                'rep_r_max': 2.0,
                'rep_r_cut': 6,
                'dt': 0.001,
                'tau': 0.1,
                'run_step': 1.5e8,
                'dump_period': 1e5,
                'log_period': 1e4,
            }
            job = project.open_job(sp)
            job.init()