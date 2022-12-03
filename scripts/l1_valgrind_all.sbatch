#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH -o slurm-jobs/l1-valgring_all.out
#SBATCH --exclusive
#SBATCH -p slurm-a

mkdir -p xl1
module load armpl/21.0.0 gnu/11.2.0
spl gcc@12.2.0

fn="xl1/noblocking"
make -B stencil OPTION="-DNOBS -DDIMM=1000"
./stencil > $fn.out 2>&1
./stencil > $fn.out 2>&1
python3 scripts/assert.py -p small -o $fn.out > $fn.ass 2>&1
valgrind --tool=cachegrind ./stencil > $fn.valgrind 2>&1

for i in $(seq 2 2 100); do
    fn="xl1/$i-$i-0"
    make -B stencil OPTION="-DBSZ=$i -DBSY=$i -DDIMM=1000"
    ./stencil > $fn.out 2>&1
    ./stencil > $fn.out 2>&1
    python3 scripts/assert.py -p small -o $fn.out > $fn.ass 2>&1
    valgrind --tool=cachegrind ./stencil > $fn.valgrind 2>&1

    fn="xl1/0-$i-$i"
    make -B stencil OPTION="-DBSX=$i -DBSY=$i -DDIMM=1000"
    ./stencil > $fn.out 2>&1
    ./stencil > $fn.out 2>&1
    python3 scripts/assert.py -p small -o $fn.out > $fn.ass 2>&1
    valgrind --tool=cachegrind ./stencil > $fn.valgrind 2>&1

    fn="xl1/$i-$i-$i"
    make -B stencil OPTION="-DBSZ=$i -DBSX=$i -DBSY=$i -DDIMM=1000"
    ./stencil > $fn.out 2>&1
    ./stencil > $fn.out 2>&1
    python3 scripts/assert.py -p small -o $fn.out > $fn.ass 2>&1
    valgrind --tool=cachegrind ./stencil > $fn.valgrind 2>&1
done

