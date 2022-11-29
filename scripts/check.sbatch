#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH -o slurm_jobs/job%J.out
#SBATCH --exclusive
#SBATCH -p basic-slurm

# Run check script
python3 scripts/speedup.py -r 500 500 500 5

