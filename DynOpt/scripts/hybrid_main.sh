#!/bin/bash

dims=(2 5 10 20 50 100)
zfactors=0.01,0.1,1.0,10.0
#zfactors=0.1,0.5,1.0,2.0

# ----------------------------------------------------------------------------


pred1="hybrid-autoregressive-rnn"
pred2="rnn"
algnameaddition1="" 
algnameaddition2=""


#pred1="tcn"
#pred2="tcn"
#algnameaddition1="_auto_predRND" # for autoTCN: _auto_ ... !!! (that is TCN with uncertainty estimate
#algnameaddition2="_auto_predUNC" 								

#pred1="tcn"
#pred2="tcn"
#algnameaddition1="_predRND" 
#algnameaddition2="_predDEV"

#pred1="autoregressive"
#pred2="autoregressive"
#algnameaddition1="_predRND" 
#algnameaddition2="_predDEV"

 								
useuncs1="False"
useuncs2="False"
reinimode1="pred-RND"
reinimode2="pred-RND"


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
	#./subscript.job "$pred1" "$algnameaddition1" "$useuncs1" "$reinimode1" "$zfactors" "$d" &
	#./subscript.job "$pred2" "$algnameaddition2" "$useuncs2" "$reinimode2" "$zfactors" "$d" &
	# SRR
	# sbatch --mem=16G --job-name="d$d-rnd" --output="slurm_d$d-rnd.%j.out" --error="slurm_d$d-rnd.%j.err" icann_subscript.job "$pred1" "$algnameaddition1" "$useuncs1" "$reinimode1" "$zfactors" "$d" &
	# sbatch --mem=16G --job-name="d$d-unc" --output="slurm_d$d-unc.%j.out" --error="slurm_d$d-unc.%j.err" icann_subscript.job "$pred2" "$algnameaddition2" "$useuncs2" "$reinimode2" "$zfactors" "$d" &
	sbatch --mem=16G --job-name="d$d-hybr" --output="slurm_d$d-hybr.%j.out" --error="slurm_d$d-hybr.%j.err" hybrid_subscript.job "$pred1" "$algnameaddition1" "$useuncs1" "$reinimode1" "$zfactors" "$d" &
	sbatch --mem=16G --job-name="d$d-rnn" --output="slurm_d$d-rnn.%j.out" --error="slurm_d$d-rnn.%j.err" hybrid_subscript.job "$pred2" "$algnameaddition2" "$useuncs2" "$reinimode2" "$zfactors" "$d" &
	# MPB?	
	#sbatch --mem=16G --job-name="d$d-mpbkal" --output="slurm_d$d-mpbkal.%j.out" --error="slurm_d$d-mpbkal.%j.err" icann_subscript.job "$pred1" "$algnameaddition1" "$useuncs1" "$reinimode1" "$zfactors" "$d" &
	#sbatch --mem=16G --job-name="d$d-mpbunc" --output="slurm_d$d-mpbunc.%j.out" --error="slurm_d$d-mpbunc.%j.err" icann_subscript.job "$pred2" "$algnameaddition2" "$useuncs2" "$reinimode2" "$zfactors" "$d" &
done


#./subscript.job "$pred1" "$algnameaddition1" "$useuncs1" "$reinimode1" "$zfactors" &
#./subscript.job "$pred2" "$algnameaddition2" "$useuncs2" "$reinimode2" "$zfactors" &

#sbatch --mem=16G --job-name="d$d-hybr" --output="slurm_d$d-hybr.%j.out" --error="slurm_d$d-hybr.%j.err" hybrid_subscript.job "$pred1" "$algnameaddition1" "$useuncs1" "$reinimode1" "$zfactors" &
#sbatch --mem=16G --job-name="d$d-rnn" --output="slurm_d$d-rnn.%j.out" --error="slurm_d$d-rnn.%j.err" hybrid_subscript.job "$pred2" "$algnameaddition2" "$useuncs2" "$reinimode2" "$zfactors" &
	
wait
