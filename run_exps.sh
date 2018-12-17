#!/bin/bash

for K in 32 64 ; do
    for L in 2 4 8 16 ; do
        ./py-sbatch.sh -m ./hw2/experiments.py -K ${K} -L ${L} -n "exp1_1_K${K}_L${L}"
    done
done
