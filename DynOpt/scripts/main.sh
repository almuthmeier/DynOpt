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
#epuncfactor=0.0
#./subscript2.job "$pred1" "$algnameaddition1" "$useuncs1" "$epuncfactor" "$d" &
#./subscript2.job "$pred2" "$algnameaddition2" "$useuncs2" "$epuncfactor" "$d" &

# ----------------------------------------------------------------------------
# tcn, autotcn

pred1="tcn"
pred2="tcn"
algnameaddition1=""								# TCN
algnameaddition2a="_auto_dynsig"				# AutoTCN variants
algnameaddition2b="_auto_36-2"
useuncs1="False"
useuncs2="True"
epuncfactor1=0.0
epuncfactor2a=-1 #dynamic
epuncfactor2b=0.47 #(36.16%)



# for sphere
## normal
#sbatch --mem=35G --job-name="d$d-tcn" --output="slurm_d$d-tcn.%j.out" --error="slurm_d$d-tcn.%j.err" subscript2.job "$pred1" "$algnameaddition1" "$useuncs1" "$epuncfactor1" "$d" &
#sbatch --mem=65G --job-name="d$d-auto" --output="slurm_d$d-auto.%j.out" --error="slurm_d$d-auto.%j.err" subscript2.job "$pred2" "$algnameaddition2" "$useuncs2" "$epuncfactor2" "$d" &

## different sigmas
#sbatch --mem=45G --job-name="d$d-a-auto" --output="slurm_d$d-a-auto.%j.out" --error="slurm_d$d-a-auto.%j.err" subscript2.job "$pred2" "$algnameaddition2a" "$useuncs2" "$epuncfactor2a" "$d" &
#sbatch --mem=45G --job-name="d$d-b-auto" --output="slurm_d$d-b-auto.%j.out" --error="slurm_d$d-b-auto.%j.err" subscript2.job "$pred2" "$algnameaddition2b" "$useuncs2" "$epuncfactor2b" "$d" &

# for mpb (different jobnames)
## normal			
#sbatch --mem=35G --job-name="Md$d-tcn" --output="slurm_mpbcorr_d$d-tcn.%j.out" --error="slurm_mpbcorr_d$d-tcn.%j.err" subscript2.job "$pred1" "$algnameaddition1" "$useuncs1" "$epuncfactor1" "$d" &
#sbatch --mem=65G --job-name="Md$d-auto" --output="slurm_mpbcorr_d$d-auto.%j.out" --error="slurm_mpbcorr_d$d-auto.%j.err" subscript2.job "$pred2" "$algnameaddition2" "$useuncs2" "$epuncfactor2" "$d" &

## different sigmas
sbatch --mem=45G --job-name="Md$d-a-auto" --output="slurm_mpbcorr_d$d-a-auto.%j.out" --error="slurm_mpbcorr_d$d-a-auto.%j.err" subscript2.job "$pred2" "$algnameaddition2a" "$useuncs2" "$epuncfactor2a" "$d" &
sbatch --mem=45G --job-name="Md$d-b-auto" --output="slurm_mpbcorr_d$d-b-auto.%j.out" --error="slurm_mpbcorr_d$d-b-auto.%j.err" subscript2.job "$pred2" "$algnameaddition2b" "$useuncs2" "$epuncfactor2b" "$d" &
# ----------------------------------------------------------------------------


for d in "${dims[@]}"
do
	sbatch --mem=45G --job-name="d$d-dyn-auto" --output="slurm_d$d-dyn-auto.%j.out" --error="slurm_d$d-dyn-auto.%j.err" subscript2.job "$pred2" "$algnameaddition2a" "$useuncs2" "$epuncfactor2a" "$d" &
	sbatch --mem=45G --job-name="d$d-047-auto" --output="slurm_d$d-047-auto.%j.out" --error="slurm_d$d-047-auto.%j.err" subscript2.job "$pred2" "$algnameaddition2b" "$useuncs2" "$epuncfactor2b" "$d" &
done

wait
