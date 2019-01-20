#!/bin/bash

dims=(2 5 10 20)


# ----------------------------------------------------------------------------
# no, ar

#pred1="no"
#pred2="autoregressive"
#algnameaddition1=""
#algnameaddition2=""
#useuncs1="False"
#useuncs2="False"
#./subscript2.job "$pred1" "$algnameaddition1" "$useuncs1" "$d" &
#./subscript2.job "$pred2" "$algnameaddition2" "$useuncs2" "$d" &

# ----------------------------------------------------------------------------
# tcn, autotcn

pred1="tcn"
pred2="tcn"
algnameaddition1=""
algnameaddition2="_auto"
useuncs1="False"
useuncs2="True"

#sbatch --mem=35G --job-name="d$d-tcn" --output="slurm_d$d-tcn.%j.out" --error="slurm_d$d-tcn.%j.err" subscript2.job "$pred1" "$algnameaddition1" "$useuncs1" "$d" &
#sbatch --mem=65G --job-name="d$d-auto" --output="slurm_d$d-auto.%j.out" --error="slurm_d$d-auto.%j.err" subscript2.job "$pred2" "$algnameaddition2" "$useuncs2" "$d" &			

# ----------------------------------------------------------------------------


for d in "${dims[@]}"
do
	sbatch --mem=35G --job-name="Md$d-tcn" --output="slurm_mpbcorr_d$d-tcn.%j.out" --error="slurm_mpbcorr_d$d-tcn.%j.err" subscript2.job "$pred1" "$algnameaddition1" "$useuncs1" "$d" &
	sbatch --mem=65G --job-name="Md$d-auto" --output="slurm_mpbcorr_d$d-auto.%j.out" --error="slurm_mpbcorr_d$d-auto.%j.err" subscript2.job "$pred2" "$algnameaddition2" "$useuncs2" "$d" &
done

wait
