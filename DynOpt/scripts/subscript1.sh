#!/bin/bash

bsizes=(8 16 32)
for bs in "${bsizes[@]}"
do
	sbatch --job-name="lr-$1-bs-$bs" --output="slurm_lr-$1-bs-$bs.%j.out" --error="slurm_lr-$1-bs-$bs.%j.err" subscript2.job $1 $bs &
done

wait