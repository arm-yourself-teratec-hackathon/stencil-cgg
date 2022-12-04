#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH -o job-best64c-%J.out
#SBATCH --exclusive
#SBATCH -p slurm-a

module load gnu/11.2.0

make -B stencil OPTION="-DBSZ2=2 -DBSY2=2 -DDIMM=1000"
for i in $(seq 1 1 10); do
    ./stencil > run$i.out 2>&1
done

spl gcc@12.2.0

make -B stencil LDFLAGS= OPTION="-DBSZ2=2 -DBSY2=2 -DDIMM=1000"
for i in $(seq 1 1 10); do
    ./stencil > _run$i.out 2>&1
done
