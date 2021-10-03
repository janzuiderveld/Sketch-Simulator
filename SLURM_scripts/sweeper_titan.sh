#!/bin/bash
#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --gres=gpu:1
#SBATCH -t 5:00:00
#SBATCH --job-name=GPUer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3


cd /home/janz/sketch_sim
source sketch_env/bin/activate

sweep_config=$1

echo "Running with config: $sweep_config"

sweep_config_path=/home/janz/sketch_sim/sweep_configs/"${sweep_config}" 2>&1

echo "Sweep config path: $sweep_config_path"

SWEEP_ID=$(/home/janz/.local/bin/wandb sweep -p Sketch-sim "${sweep_config_path}" 2>&1)

echo "Sweep ID: $SWEEP_ID"

/home/janz/.local/bin/wandb agent janzuiderveld/Sketch-sim/"${SWEEP_ID: -8}"