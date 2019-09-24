#!/bin/bash

dims=(2 5 10 20)
#dims=(10)

# ----------------------------------------------------------------------------


#predictor

pred1="tcn"
pred2="no"

cmavariant1="predcma_external"
cmavariant2="predcma_internal"

predvariant1="d"
predvariant2="branke"
							
useuncs1="True"
useuncs2="False"


algnameaddition1="_$predvariant1"
algnameaddition2="_$predvariant2"

# ----------------------------------------------------------------------------
# no, ar (Vegas)


#./subscript.job "$pred1" "$algnameaddition1" "$useuncs1" "$reinimode1" "$zfactors" "$d" &
#./subscript.job "$pred2" "$algnameaddition2" "$useuncs2" "$reinimode2" "$zfactors" "$d" &

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# tcn, autotcn (Nevada)



# for sphere
## normal
#sbatch --mem=35G --job-name="d$d-tcn" --output="slurm_d$d-tcn.%j.out" --error="slurm_d$d-tcn.%j.err" subscript.job "$pred1" "$algnameaddition1" "$useuncs1" "$reinimode" "$zfactors" "$d" &
#sbatch --mem=65G --job-name="d$d-auto" --output="slurm_d$d-auto.%j.out" --error="slurm_d$d-auto.%j.err" subscript.job "$pred2" "$algnameaddition2" "$useuncs2" "$reinimode" "$zfactors" "$d" &

## different sigmas
#sbatch --mem=45G --job-name="d$d-a-auto" --output="slurm_d$d-a-auto.%j.out" --error="slurm_d$d-a-auto.%j.err" subscript.job "$pred2" "$algnameaddition2a" "$useuncs2" "$reinimode" "$zfactors" "$d" &
#sbatch --mem=45G --job-name="d$d-b-auto" --output="slurm_d$d-b-auto.%j.out" --error="slurm_d$d-b-auto.%j.err" subscript.job "$pred2" "$algnameaddition2b" "$useuncs2" "$reinimode" "$zfactors" "$d" &

#sbatch --mem=45G --job-name="d$d-oalskal" --output="slurm_d$d-oalskal.%j.out" --error="slurm_d$d-oalskal.%j.err" subscript.job "$pred2" "$algnameaddition2a" "$useuncs2" "$reinimode" "$zfactors" "$d" &
#sbatch --mem=45G --job-name="d$d-rmse_auto" --output="slurm_d$d-rmse_auto.%j.out" --error="slurm_d$d-rmse_auto.%j.err" subscript.job "$pred2" "$algnameaddition2b" "$useuncs2" "$reinimode" "$zfactors" "$d" &

# 24.1. (pred-DEV reinitialization type)
#sbatch --mem=35G --job-name="d$d-tcn_predDEV" --output="slurm_d$d-tcn_predDEV.%j.out" --error="slurm_d$d-tcn_predDEV.%j.err" subscript.job "$pred1" "$algnameaddition1" "$useuncs1" "$reinimode" "$zfactors" "$d" &

#sbatch --mem=24G --job-name="d$d-t_RND" --output="slurm_d$d-t_RND.%j.out" --error="slurm_d$d-t_RND.%j.err" subscript.job "$pred1" "$algnameaddition1" "$useuncs1" "$reinimode1" "$zfactors" "$d" &
#sbatch --mem=24G --job-name="d$d-t_DEV" --output="slurm_d$d-t_DEV.%j.out" --error="slurm_d$d-t_RND.%j.err" subscript.job "$pred2" "$algnameaddition2" "$useuncs2" "$reinimode2" "$zfactors" "$d" &


# for mpb (different jobnames)
## normal			
#sbatch --mem=35G --job-name="Md$d-tcn" --output="slurm_mpbcorr_d$d-tcn.%j.out" --error="slurm_mpbcorr_d$d-tcn.%j.err" subscript.job "$pred1" "$algnameaddition1" "$useuncs1" "$reinimode" "$zfactors" "$d" &
#sbatch --mem=65G --job-name="Md$d-auto" --output="slurm_mpbcorr_d$d-auto.%j.out" --error="slurm_mpbcorr_d$d-auto.%j.err" subscript.job "$pred2" "$algnameaddition2" "$useuncs2" "$reinimode" "$zfactors" "$d" &

## different sigmas
#sbatch --mem=45G --job-name="Md$d-a-auto" --output="slurm_mpbcorr_d$d-a-auto.%j.out" --error="slurm_mpbcorr_d$d-a-auto.%j.err" subscript.job "$pred2" "$algnameaddition2a" "$useuncs2" "$reinimode" "$zfactors" "$d" &
#sbatch --mem=45G --job-name="Md$d-b-auto" --output="slurm_mpbcorr_d$d-b-auto.%j.out" --error="slurm_mpbcorr_d$d-b-auto.%j.err" subscript.job "$pred2" "$algnameaddition2b" "$useuncs2" "$reinimode" "$zfactors" "$d" &

#sbatch --mem=45G --job-name="d$d-001-auto" --output="slurm_d$d-001-auto.%j.out" --error="slurm_d$d-001-auto.%j.err" subscript.job "$pred2" "$algnameaddition2a" "$useuncs2" "$reinimode" "$zfactors" "$d" &
#sbatch --mem=45G --job-name="d$d-01-auto" --output="slurm_d$d-01-auto.%j.out" --error="slurm_d$d-01-auto.%j.err" subscript.job "$pred2" "$algnameaddition2b" "$useuncs2" "$reinimode" "$zfactors" "$d" &

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


for d in "${dims[@]}"
do
	#./subscript.job "$pred1" "$algnameaddition1" "$useuncs1" "$cmavariant1" "$predvariant1" "$d" &
	#./subscript.job "$pred2" "$algnameaddition2" "$useuncs2" "$cmavariant2" "$predvariant2" "$d" &	
		sbatch --mem=16G --job-name="d$d-$predvariant1" --output="slurm_d$d-$predvariant1.%j.out" --error="slurm_d$d-$predvariant1.%j.err" subscript.job "$pred1" "$algnameaddition1" "$useuncs1" "$cmavariant1" "$predvariant1" "$d" &
	#	sbatch --mem=16G --job-name="d$d-$predvariant2" --output="slurm_d$d-$predvariant2.%j.out" --error="slurm_d$d-$predvariant2.%j.err" subscript.job "$pred2" "$algnameaddition2" "$useuncs2" "$cmavariant2" "$predvariant2" "$d" &
done



	
wait
