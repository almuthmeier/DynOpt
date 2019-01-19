#!/bin/bash

dims=(2 5 10 20)




pred="tcn"
algnameaddition1=""
algnameaddition2="_auto"

for d in "${dims[@]}"
do
	sbatch --job-name="d$d-tcn" --output="slurm_d$d-auto.%j.out" --error="slurm_d$d-auto.%j.err" subscript2.job $pred $algnameaddition1 $d &
	sbatch --job-name="d$d-auto" --output="slurm_d$d-auto.%j.out" --error="slurm_d$d-auto.%j.err" subscript2.job $pred $algnameaddition2 $d &
done

wait