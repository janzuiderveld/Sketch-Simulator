#!/bin/bash
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --job-name=GPUer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3

cd /home/janz/sketch_sim
source sketch_env/bin/activate

python3 $1