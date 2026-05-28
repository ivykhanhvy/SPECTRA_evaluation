#!/bin/bash
#SBATCH -c 1  # Number of Cores per MPI Task
#SBATCH -N 1 # Number of Nodes
#SBATCH --mem=16G  # Requested Memory Per Node
#SBATCH -p cpu  # Partition
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

module load python/3.12.3
module load conda/latest

conda activate chemConda
python3 classical_model_baselines.py 
