#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH -o slurm-jobs/check.out
#SBATCH --exclusive
#SBATCH -p basic-slurm

# Run check script
python3 scripts/speedup.py -m
