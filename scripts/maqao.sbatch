#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH -o slurm-jobs/maqao.out
#SBATCH --exclusive
#SBATCH -p basic-slurm

# Run MAQAO
maqao oneview -R1 --config=cfg.lua -xp=maqao-arm --replace
