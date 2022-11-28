#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH -o job.%J.out
#SBATCH  --exclusive
#SBATCH  -p big

# To submit the script run the command sbatch stencil-big-test.sbatch
# big in the line above is the Slurm queue name, please check it is correct.

./stencil 1000 1000 1000 10
