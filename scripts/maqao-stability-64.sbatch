#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH -o slurm-jobs/maqao-stability-64.out
#SBATCH --exclusive
#SBATCH --priority=1
#SBATCH -p slurm-a

# Run MAQAO
$HOME/maqao.aarch64.2.16.0/maqao.aarch64 oneview -S1 --config=cfg_stability.lua -xp=maqao-stability-64 --replace
