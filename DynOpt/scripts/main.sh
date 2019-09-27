#!/bin/bash

dims=(2 5 10 20)
dims=(20)

# ----------------------------------------------------------------------------


#predictor

pred1="no"
pred2="no"
pred3="no"
pred4="no"
pred5="no"
pred6="tcn"
pred7="tcn"
pred8="tcn"
pred9="truepred"
pred10="truepred"
pred11="truepred"

cmavariant1="static"
cmavariant2="resetcma"
cmavariant3="predcma_internal"
cmavariant4="predcma_internal"
cmavariant5="predcma_internal"
cmavariant6="predcma_external"
cmavariant7="predcma_external"
cmavariant8="predcma_external"
cmavariant9="predcma_external"
cmavariant10="predcma_external"
cmavariant11="predcma_external"

predvariant1="None"
predvariant2="None"
predvariant3="branke"
predvariant4="hdwom"
predvariant5="hd"
predvariant6="d"
predvariant7="c"
predvariant8="a"
predvariant9="a"
predvariant10="c"
predvariant11="d"
							
useuncs1="False"
useuncs2="False"
useuncs3="False"
useuncs4="False"
useuncs5="False"
useuncs6="True"
useuncs7="True"
useuncs8="True"
useuncs9="True"
useuncs10="True"
useuncs11="True"


algnameaddition1="_static_$predvariant1"
algnameaddition2="_$predvariant2"
algnameaddition3="_$predvariant3"
algnameaddition4="_$predvariant4"
algnameaddition5="_$predvariant5"
algnameaddition6="_$predvariant6"
algnameaddition7="_$predvariant7"
algnameaddition8="_$predvariant8"
algnameaddition9="_$predvariant9"
algnameaddition10="_$predvariant10"
algnameaddition11="_$predvariant11"

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
	#	sbatch --mem=16G --gres=gpu:0 --job-name="d$d-$predvariant1" --output="slurm_d$d-$predvariant1.%j.out" --error="slurm_d$d-$predvariant1.%j.err" subscript.job "$pred1" "$algnameaddition1" "$useuncs1" "$cmavariant1" "$predvariant1" "$d" &
	#	sbatch --mem=16G --gres=gpu:0 --job-name="d$d-$predvariant2" --output="slurm_d$d-$predvariant2.%j.out" --error="slurm_d$d-$predvariant2.%j.err" subscript.job "$pred2" "$algnameaddition2" "$useuncs2" "$cmavariant2" "$predvariant2" "$d" &
	#	sbatch --mem=16G --gres=gpu:0 --job-name="d$d-$predvariant3" --output="slurm_d$d-$predvariant3.%j.out" --error="slurm_d$d-$predvariant3.%j.err" subscript.job "$pred3" "$algnameaddition3" "$useuncs3" "$cmavariant3" "$predvariant3" "$d" &
	#	sbatch --mem=16G --gres=gpu:0 --job-name="d$d-$predvariant4" --output="slurm_d$d-$predvariant4.%j.out" --error="slurm_d$d-$predvariant4.%j.err" subscript.job "$pred4" "$algnameaddition4" "$useuncs4" "$cmavariant4" "$predvariant4" "$d" &
	#	sbatch --mem=16G --gres=gpu:0 --job-name="d$d-$predvariant5" --output="slurm_d$d-$predvariant5.%j.out" --error="slurm_d$d-$predvariant5.%j.err" subscript.job "$pred5" "$algnameaddition5" "$useuncs5" "$cmavariant5" "$predvariant5" "$d" &
	#	sbatch --mem=16G --gres=gpu:1 --job-name="d$d-$predvariant6" --output="slurm_d$d-$predvariant6.%j.out" --error="slurm_d$d-$predvariant6.%j.err" subscript.job "$pred6" "$algnameaddition6" "$useuncs6" "$cmavariant6" "$predvariant6" "$d" &
	#	sbatch --mem=16G --gres=gpu:1 --job-name="d$d-$predvariant7" --output="slurm_d$d-$predvariant7.%j.out" --error="slurm_d$d-$predvariant7.%j.err" subscript.job "$pred7" "$algnameaddition7" "$useuncs7" "$cmavariant7" "$predvariant7" "$d" &
	#	sbatch --mem=16G --gres=gpu:1 --job-name="d$d-$predvariant8" --output="slurm_d$d-$predvariant8.%j.out" --error="slurm_d$d-$predvariant8.%j.err" subscript.job "$pred8" "$algnameaddition8" "$useuncs8" "$cmavariant8" "$predvariant8" "$d" &
	#	sbatch --mem=16G --gres=gpu:0 --job-name="d$d-$predvariant9" --output="slurm_d$d-$predvariant9.%j.out" --error="slurm_d$d-$predvariant9.%j.err" subscript.job "$pred9" "$algnameaddition9" "$useuncs9" "$cmavariant9" "$predvariant9" "$d" &
	#	sbatch --mem=16G --gres=gpu:0 --job-name="d$d-$predvariant10" --output="slurm_d$d-$predvariant10.%j.out" --error="slurm_d$d-$predvariant10.%j.err" subscript.job "$pred10" "$algnameaddition10" "$useuncs10" "$cmavariant10" "$predvariant10" "$d" &
		sbatch --mem=16G --gres=gpu:0 --job-name="d$d-$predvariant11" --output="slurm_d$d-$predvariant11.%j.out" --error="slurm_d$d-$predvariant11.%j.err" subscript.job "$pred11" "$algnameaddition11" "$useuncs11" "$cmavariant11" "$predvariant11" "$d" &		
done



	
wait
