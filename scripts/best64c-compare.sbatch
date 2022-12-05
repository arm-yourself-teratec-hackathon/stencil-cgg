#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH -o job-best64c-%J.out
#SBATCH --exclusive
#SBATCH -p slurm-a

# source $HOME/.aliases
# spl gcc@12.2.0

make -B
for i in $(seq 1 1 10); do
    ./stencil > 1_run$i.out 2>&1
    ./stencil > 2_run$i.out 2>&1
done

# cp stencil2.cxx src/stencil.cxx

# make -B stencil LDFLAGS= OPTION="-DBSZ2=2 -DBSY2=2 -DDIMM=1000"
# for i in $(seq 1 1 10); do
#     ./stencil > 2_run$i.out 2>&1
# done

# cp stencil3.cxx src/stencil.cxx

# make -B stencil LDFLAGS= OPTION="-DBSZ2=2 -DBSY2=2 -DDIMM=1000"
# for i in $(seq 1 1 10); do
#     ./stencil > 3_run$i.out 2>&1
# done
