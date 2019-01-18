#!/bin/bash

kernelsizes=(2 3 4 5 6 7)
for k in "${kernelsizes[@]}"
do
	sbatch --job-name="ks-$k" --output="slurm_ks-$k.%j.out" --error="slurm_ks-$k.%j.err" subscript.job $k &
done

wait
