#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH -o slurm-jobs/small.out
#SBATCH --exclusive
#SBATCH -p basic-slurm

# To submit the script run the command sbatch stencil-big-test.sbatch
# small in the line above is the Slurm Queue name, please confirm it is correct.

# Run small test
./stencil 100 100 100 5
