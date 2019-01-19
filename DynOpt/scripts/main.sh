#!/bin/bash

dims=(2 5 10 20)




pred1="no"
pred2="autoregressive"
algnameaddition1=""
algnameaddition2=""

#sbatch --job-name="d$d-tcn" --output="slurm_d$d-tcn.%j.out" --error="slurm_d$d-tcn.%j.err" subscript2.job "$pred1" "$algnameaddition1" "$d" &
#sbatch --job-name="d$d-auto" --output="slurm_d$d-auto.%j.out" --error="slurm_d$d-auto.%j.err" subscript2.job "$pred2" "$algnameaddition2" "$d" &

for d in "${dims[@]}"
do
	./subscript2.job "$pred1" "$algnameaddition1" "$d" &
	./subscript2.job "$pred2" "$algnameaddition2" "$d" &
done

wait