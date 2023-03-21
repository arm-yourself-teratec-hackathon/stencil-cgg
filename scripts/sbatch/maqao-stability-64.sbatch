#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH -o maqao_stability_64-job%J.out
#SBATCH --exclusive
#SBATCH --priority=1
#SBATCH -p slurm-a

# Run MAQAO

if [ "$(pwd | sed "s|.*/||g")" != "Stencil_CGG" ]; then
    echo "script must be runned from base dir Stencil_CGG" && return 0
fi

$HOME/maqao.aarch64.2.16.0/maqao.aarch64 oneview -S1 --config=scripts/cfg_stability.lua -xp=maqao-stability-64 --replace
