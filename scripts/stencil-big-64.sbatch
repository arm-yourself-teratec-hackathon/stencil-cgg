#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH -o stencil-64c-big.out
#SBATCH --exclusive
#SBATCH --priority=1
#SBATCH -p slurm-a

./stencil > /dev/null # Warm up
./stencil # Real code
