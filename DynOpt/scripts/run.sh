#!/bin/bash

# damit dieses Skript aus Eclipse heraus ausführbar wird:
# https://askubuntu.com/questions/514247/running-a-bash-file-in-eclipse (11.5.18)
#	- Skript ausführbar machen mit: chmod +x run.sh
#	- Run -> External Tools -> External Tool Configurations
#		- Doppelklick auf "Program"
#		- Location: nach dem Skript suchen
#		- Working Directory: Verzeichnis, in dem das Skript liegt suchen

#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------

# benchmark problem
algorithm="dynpso" 			# dynea or dynpso
repetitions=1
chgperiods=10
lenchgperiod=20
ischgperiodrandom=False
benchmarkfunction=sphere	# sphere, rosenbrock, rastrigin, mpbnoisy, mpbrandom (neu)
							# defines the benchmark function, must be located 
							# in the datasets folder of this project [for the dataset as input]
benchmarkfunctionfolder=GECCO2018 # parent directory of the benchmark functions
								   # and child directory of the datasets folder
								   # of this project [for the dataset as input]
outputdirectorypath="/home/ameier/Documents/Promotion/GIT_Lab/DynOptimization/DynOpt/output/myexperiments/ff_sphere_1/"		# path to output folder

# run only some experiments of all for the benchark problem
poschgtype="linear"
fitchgtype="none"
dim=2
noise=0.0

# PSO
c1=1.496180
c2=1.496180
c3=1.496180
insertpred=False			# False, True
adaptivec3=False			# False, True
nparticles=100				#

# EA
mu=5
la=10
ro=2
mean=0.0
sigma=1.0
trechenberg=5
tau=0.5

# predictor
predictor=no				# no, rnn, autoregressive
timesteps=7

# ANN predictor
neuronstype=fixed20			# fixed20, dyn1.3
epochs=30
batchsize=1					# ist 1, weil man ja immer nur ein neues Trainingselement hat (aber beim 1. Mal könnte man ja mit größerer batch size trainineren!?!)
ngpus=1    					#


#------------------------------------------------------------------------------
# Command
#------------------------------------------------------------------------------
# es muss immer ein Leerzeichen zwischen dem Argument und dem Backslash sein!!!!!

~/.virtualenvs/promotion/prototype/bin/python3.5 ../code/input_parser.py -algorithm="$algorithm" \
-repetitions="$repetitions" \
-chgperiods="$chgperiods" \
-lenchgperiod="$lenchgperiod" \
-ischgperiodrandom="$ischgperiodrandom" \
-benchmarkfunction="$benchmarkfunction" \
-benchmarkfunctionfolder="$benchmarkfunctionfolder" \
-outputdirectorypath="$outputdirectorypath" \
-poschgtype="$poschgtype" \
-fitchgtype="$fitchgtype" \
-dim="$dim" \
-noise="$noise" \
-c1="$c1" \
-c2="$c2" \
-c3="$c3" \
-insertpred="$insertpred" \
-adaptivec3="$adaptivec3" \
-nparticles="$nparticles" \
-mu="$mu" \
-la="$la" \
-ro="$ro" \
-mean="$mean" \
-sigma="$sigma" \
-trechenberg="$trechenberg" \
-tau="$tau" \
-predictor="$predictor" \
-timesteps="$timesteps" \
-neuronstype="$neuronstype" \
-epochs="$epochs" \
-batchsize="$batchsize" \
-ngpus="$ngpus" \
