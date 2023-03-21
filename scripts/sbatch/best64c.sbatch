#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH -o job-best64c-%J.out
#SBATCH --exclusive
#SBATCH -p slurm-a

make -B stencil OPTION="-DBSZ2=2 -DBSY2=2 -DDIMM=1000"
./stencil > /dev/null # Warm-up run
./stencil
