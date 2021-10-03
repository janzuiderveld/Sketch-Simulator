#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH --job-name=CPUer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /home/janz/sketch_sim
source sketch_env/bin/activate

python3 $1