#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH -o slurm-jobs/l1-valgring_all.out
#SBATCH --exclusive
#SBATCH -p slurm-a

mkdir -p xl
spl gcc@12.2.0

fn="xl/noblocking"
make -B stencil OPTION="-DNOBS -DDIMM=1000"
./stencil > $fn.out 2>&1
./stencil > $fn.out 2>&1
python3 scripts/assert.py -p big -o $fn.out > $fn.ass 2>&1
# valgrind --tool=cachegrind ./stencil > $fn.valgrind 2>&1

for i in $(seq 2 2 100); do

    # Test best runs on first loop !

    fn="xl/l1_$i-$i-0"
    make -B stencil OPTION="-DBSZ1=$i -DBSY1=$i -DDIMM=1000"
    ./stencil > $fn.out 2>&1
    ./stencil > $fn.out 2>&1
    python3 scripts/assert.py -p big -o $fn.out > $fn.ass 2>&1
    # valgrind --tool=cachegrind ./stencil > $fn.valgrind 2>&1

    fn="xl/l1_0-$i-$i"
    make -B stencil OPTION="-DBSX1=$i -DBSY1=$i -DDIMM=1000"
    ./stencil > $fn.out 2>&1
    ./stencil > $fn.out 2>&1
    python3 scripts/assert.py -p big -o $fn.out > $fn.ass 2>&1
    # valgrind --tool=cachegrind ./stencil > $fn.valgrind 2>&1

    fn="xl/l1_$i-$i-$i"
    make -B stencil OPTION="-DBSZ1=$i -DBSX1=$i -DBSY1=$i -DDIMM=1000"
    ./stencil > $fn.out 2>&1
    ./stencil > $fn.out 2>&1
    python3 scripts/assert.py -p big -o $fn.out > $fn.ass 2>&1
    # valgrind --tool=cachegrind ./stencil > $fn.valgrind 2>&1

    # Test best runs on second & main loop !

    fn="xl/l2_$i-$i-0"
    make -B stencil OPTION="-DBSZ2=$i -DBSY2=$i -DDIMM=1000"
    ./stencil > $fn.out 2>&1
    ./stencil > $fn.out 2>&1
    python3 scripts/assert.py -p big -o $fn.out > $fn.ass 2>&1
    # valgrind --tool=cachegrind ./stencil > $fn.valgrind 2>&1

    fn="xl/l2_0-$i-$i"
    make -B stencil OPTION="-DBSX2=$i -DBSY2=$i -DDIMM=1000"
    ./stencil > $fn.out 2>&1
    ./stencil > $fn.out 2>&1
    python3 scripts/assert.py -p big -o $fn.out > $fn.ass 2>&1
    # valgrind --tool=cachegrind ./stencil > $fn.valgrind 2>&1

    fn="xl/l2_$i-$i-$i"
    make -B stencil OPTION="-DBSZ2=$i -DBSX2=$i -DBSY2=$i -DDIMM=1000"
    ./stencil > $fn.out 2>&1
    ./stencil > $fn.out 2>&1
    python3 scripts/assert.py -p big -o $fn.out > $fn.ass 2>&1
    # valgrind --tool=cachegrind ./stencil > $fn.valgrind 2>&1
done

