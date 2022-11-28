#!/bin/bash
icpc -xHost -Ofast -g3 -fno-omit-frame-pointer -fopenmp -qmkl -funroll-loops stencil.cxx -o stencil
