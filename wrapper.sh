#!/bin/bash -l
#SBATCH -J dd
#SBATCH -o tmp_res/output_18_%j.txt
#SBATCH -e tmp_res/errors_18_%j.txt
#SBATCH -t 2-12:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=16000
#SBATCH

srun python minetBrats.py 
