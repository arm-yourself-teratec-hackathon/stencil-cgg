#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH -o slurm-jobs/big.out
#SBATCH --exclusive
#SBATCH --priority=1
#SBATCH -p slurm-a

./stencil > /dev/null
./stencil
