#!/bin/bash
#SBATCH -J sim_
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --constraint="a100|v100|a40"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liviaguttieres@g.harvard.edu
#SBATCH --signal=TERM@600
#SBATCH --array=0-9

# --- Define directories ---
RESULTS_DIR="results"
LOGS_DIR="${RESULTS_DIR}/slurm_logs"

# --- Define unique output and error logs ---
#SBATCH -o ${LOGS_DIR}/%x_%j_%a.out
#SBATCH -e ${LOGS_DIR}/%x_%j_%a.err

if [ -z "$REPLICA" ]; then
    echo "❌ ERROR: Environment variable REPLICA is not set."
    echo "Usage: sbatch --export=ALL,REPLICA=<number>,MODE=<opt|range> replicas_combined.sh"
    exit 1
fi
REPLICA_ID="$REPLICA"

if [ -z "$MODE" ]; then
    echo "❌ ERROR: Environment variable MODE is not set."
    echo "Usage: sbatch --export=ALL,REPLICA=<number>,MODE=<opt|range> replicas_combined.sh"
    exit 1
fi

# === Load environment ===
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01

# === Tell JAX where to find CUDA ===
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/n/sw/helmod-rocky8/apps/Core/cuda/12.4.1-fasrc01/cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# === Activate conda environment ===
source "/n/home02/lguttieres/miniconda3/etc/profile.d/conda.sh"
export PATH="/n/home02/lguttieres/miniconda3/condabin:${PATH}"
conda activate clean_jaxmd

# === Move to working directory ===
cd /n/brenner_lab/Lab/rotation_livia/tuning-colloidal-reactions

# === Ensure output directories exist ===
mkdir -p "${LOGS_DIR}"

# --- Map SLURM_ARRAY_TASK_ID to parameter line ---
PARAM_FILE="opt15.txt"
line_num=$((SLURM_ARRAY_TASK_ID + 1))

# --- Parse the line --- 
line=$(awk "NR==${line_num}" "$PARAM_FILE")
if [ -z "$line" ]; then
    echo "❌ No line found in ${PARAM_FILE} for task ID ${SLURM_ARRAY_TASK_ID} (line ${line_num})"
    exit 1
fi

line_clean=${line//[[:space:]]/}
IFS=',' read -r -a fields <<< "$line_clean"
if [ "${#fields[@]}" -ne 6 ]; then
    echo "❌ Error parsing ${PARAM_FILE} on line ${line_num}"
    exit 1
fi

morse_eps=${fields[2]}
kT=${fields[3]}
kT_init=${fields[4]}
percent=${fields[5]}

echo "✅ Task ID ${SLURM_ARRAY_TASK_ID} will run Line #${line_num} for Replica #${REPLICA_ID} in mode $MODE"
echo "Parameters: kT=$kT, ε=$morse_eps, percent=$percent, mode=$MODE"

# === Run the Python script ===
python3 simulate.py \
  --kT "$kT" \
  --eps "$morse_eps" \
  --percent "$percent" \
  --kT_init "$kT_init" \
  --replica_id "$REPLICA_ID" \
  --n_steps 2000000 \
  --mode "$MODE" \
  > "${RESULTS_DIR}/${SLURM_JOB_NAME}_percent${percent}_kT${kT}_rep${REPLICA_ID}_job${SLURM_JOB_ID}.log" 2>&1

echo "✅ Job complete: Task ${SLURM_ARRAY_TASK_ID}, Replica #${REPLICA_ID}, Mode $MODE"

#command to run the script
# sbatch --export=ALL,REPLICA=<replica_id>,MODE=<opt|range> replicas_combined.sh
