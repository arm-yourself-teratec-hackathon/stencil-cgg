#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH -o slurm-jobs/l2-valgring_all.out
#SBATCH --exclusive
#SBATCH -p slurm-a

mkdir -p xl2
module load armpl/21.0.0 gnu/11.2.0
spl gcc@12.2.0

fn="xl2/noblocking"
make -B stencil2 LDFLAGS= OPTION="-DNOBS -DDIMM=1000"
./stencil2 > $fn.out 2>&1
./stencil2 > $fn.out 2>&1
python3 scripts/assert.py -p small -o $fn.out > $fn.ass 2>&1
valgrind --tool=cachegrind ./stencil2 > $fn.valgrind 2>&1

for i in $(seq 2 2 100); do
    fn="xl2/$i-$i-0"
    make -B stencil2 LDFLAGS= OPTION="-DBSZ=$i -DBSY=$i -DDIMM=100"
    ./stencil2 > $fn.out 2>&1
    ./stencil2 > $fn.out 2>&1
    python3 scripts/assert.py -p small -o $fn.out > $fn.ass 2>&1
    valgrind --tool=cachegrind ./stencil2 > $fn.valgrind 2>&1

    fn="xl2/0-$i-$i"
    make -B stencil2 LDFLAGS= OPTION="-DBSX=$i -DBSY=$i -DDIMM=100"
    ./stencil2 > $fn.out 2>&1
    ./stencil2 > $fn.out 2>&1
    python3 scripts/assert.py -p small -o $fn.out > $fn.ass 2>&1
    valgrind --tool=cachegrind ./stencil2 > $fn.valgrind 2>&1

    fn="xl2/$i-$i-$i"
    make -B stencil2 LDFLAGS= OPTION="-DBSZ=$i -DBSX=$i -DBSY=$i -DDIMM=100"
    ./stencil2 > $fn.out 2>&1
    ./stencil2 > $fn.out 2>&1
    python3 scripts/assert.py -p small -o $fn.out > $fn.ass 2>&1
    valgrind --tool=cachegrind ./stencil2 > $fn.valgrind 2>&1
done

