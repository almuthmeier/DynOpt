#!/bin/bash

#dims=(2 5 10 20)
zfactors=0.01,0.1,1.0,10.0
epuncfactor=0.0 # unused


# ----------------------------------------------------------------------------


pred1="tcn"
#pred2="tcn"
algnameaddition1="_auto_predDEV" # for autoTCN: _auto_ ... !!!
#algnameaddition2="_auto_predUNC" 								
useuncs1="True"
#useuncs2="True"
reinimode1="pred-DEV"
#reinimode2="pred-UNC"


# ----------------------------------------------------------------------------
# no, ar


#./subscript2.job "$pred1" "$algnameaddition1" "$useuncs1" "$epuncfactor" "$reinimode1" "$zfactors" "$d" &
#./subscript2.job "$pred2" "$algnameaddition2" "$useuncs2" "$epuncfactor" "$reinimode2" "$zfactors" "$d" &

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# tcn, autotcn



# for sphere
## normal
#sbatch --mem=35G --job-name="d$d-tcn" --output="slurm_d$d-tcn.%j.out" --error="slurm_d$d-tcn.%j.err" subscript2.job "$pred1" "$algnameaddition1" "$useuncs1" "$epuncfactor1" "$reinimode" "$zfactors" "$d" &
#sbatch --mem=65G --job-name="d$d-auto" --output="slurm_d$d-auto.%j.out" --error="slurm_d$d-auto.%j.err" subscript2.job "$pred2" "$algnameaddition2" "$useuncs2" "$epuncfactor2" "$reinimode" "$zfactors" "$d" &

## different sigmas
#sbatch --mem=45G --job-name="d$d-a-auto" --output="slurm_d$d-a-auto.%j.out" --error="slurm_d$d-a-auto.%j.err" subscript2.job "$pred2" "$algnameaddition2a" "$useuncs2" "$epuncfactor2a" "$reinimode" "$zfactors" "$d" &
#sbatch --mem=45G --job-name="d$d-b-auto" --output="slurm_d$d-b-auto.%j.out" --error="slurm_d$d-b-auto.%j.err" subscript2.job "$pred2" "$algnameaddition2b" "$useuncs2" "$epuncfactor2b" "$reinimode" "$zfactors" "$d" &

#sbatch --mem=45G --job-name="d$d-oalskal" --output="slurm_d$d-oalskal.%j.out" --error="slurm_d$d-oalskal.%j.err" subscript2.job "$pred2" "$algnameaddition2a" "$useuncs2" "$epuncfactor2a" "$reinimode" "$zfactors" "$d" &
#sbatch --mem=45G --job-name="d$d-rmse_auto" --output="slurm_d$d-rmse_auto.%j.out" --error="slurm_d$d-rmse_auto.%j.err" subscript2.job "$pred2" "$algnameaddition2b" "$useuncs2" "$epuncfactor2b" "$reinimode" "$zfactors" "$d" &

# 24.1. (pred-DEV reinitialization type)
#sbatch --mem=35G --job-name="d$d-tcn_predDEV" --output="slurm_d$d-tcn_predDEV.%j.out" --error="slurm_d$d-tcn_predDEV.%j.err" subscript2.job "$pred1" "$algnameaddition1" "$useuncs1" "$epuncfactor1" "$reinimode" "$zfactors" "$d" &

#sbatch --mem=24G --job-name="d$d-t_RND" --output="slurm_d$d-t_RND.%j.out" --error="slurm_d$d-t_RND.%j.err" subscript2.job "$pred1" "$algnameaddition1" "$useuncs1" "$epuncfactor" "$reinimode1" "$zfactors" "$d" &
#sbatch --mem=24G --job-name="d$d-t_DEV" --output="slurm_d$d-t_DEV.%j.out" --error="slurm_d$d-t_RND.%j.err" subscript2.job "$pred2" "$algnameaddition2" "$useuncs2" "$epuncfactor" "$reinimode2" "$zfactors" "$d" &


# for mpb (different jobnames)
## normal			
#sbatch --mem=35G --job-name="Md$d-tcn" --output="slurm_mpbcorr_d$d-tcn.%j.out" --error="slurm_mpbcorr_d$d-tcn.%j.err" subscript2.job "$pred1" "$algnameaddition1" "$useuncs1" "$epuncfactor1" "$reinimode" "$zfactors" "$d" &
#sbatch --mem=65G --job-name="Md$d-auto" --output="slurm_mpbcorr_d$d-auto.%j.out" --error="slurm_mpbcorr_d$d-auto.%j.err" subscript2.job "$pred2" "$algnameaddition2" "$useuncs2" "$epuncfactor2" "$reinimode" "$zfactors" "$d" &

## different sigmas
#sbatch --mem=45G --job-name="Md$d-a-auto" --output="slurm_mpbcorr_d$d-a-auto.%j.out" --error="slurm_mpbcorr_d$d-a-auto.%j.err" subscript2.job "$pred2" "$algnameaddition2a" "$useuncs2" "$epuncfactor2a" "$reinimode" "$zfactors" "$d" &
#sbatch --mem=45G --job-name="Md$d-b-auto" --output="slurm_mpbcorr_d$d-b-auto.%j.out" --error="slurm_mpbcorr_d$d-b-auto.%j.err" subscript2.job "$pred2" "$algnameaddition2b" "$useuncs2" "$epuncfactor2b" "$reinimode" "$zfactors" "$d" &

#sbatch --mem=45G --job-name="d$d-001-auto" --output="slurm_d$d-001-auto.%j.out" --error="slurm_d$d-001-auto.%j.err" subscript2.job "$pred2" "$algnameaddition2a" "$useuncs2" "$epuncfactor2a" "$reinimode" "$zfactors" "$d" &
#sbatch --mem=45G --job-name="d$d-01-auto" --output="slurm_d$d-01-auto.%j.out" --error="slurm_d$d-01-auto.%j.err" subscript2.job "$pred2" "$algnameaddition2b" "$useuncs2" "$epuncfactor2b" ""$reinimode" "$zfactors" "$d" &

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

sbatch --mem=24G --job-name="d$d-a-d2-n001" --output="slurm_d$d-a-d2-n001.%j.out" --error="slurm_d$d-a-d2-n001.%j.err" subscript2.job "$pred1" "$algnameaddition1" "$useuncs1" "$epuncfactor" "$reinimode1" "$zfactors" &
#sbatch --mem=24G --job-name="d$d-a_alledim" --output="slurm_d$d-a_alledim.%j.out" --error="slurm_d$d-a_alledim.%j.err" subscript2.job "$pred2" "$algnameaddition2" "$useuncs2" "$epuncfactor" "$reinimode2" "$zfactors" &

#for d in "${dims[@]}"
#do	
	
#done

wait
