#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH -o slurm-jobs/medium.out
#SBATCH --exclusive
#SBATCH -p basic-slurm

# To submit the script run the command sbatch stencil-big-test.sbatch
# small in the line above is the Slurm Queue name, please confirm it is correct.

# Run medium test
./stencil 500 500 500 5
