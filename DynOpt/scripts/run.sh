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
lenchgperiod=20				# has to be set even if chgperiod==1 (is then number of generations
							# also in case ischgperiodrandom is True, lenchgperiod is needed, because lenchgperiod*chgperiods is the number of generations!! 
ischgperiodrandom=False
benchmarkfunction=sphere	# sphere, rosenbrock, rastrigin, mpbnoisy, mpbrandom (neu)
							# defines the benchmark function, must be located 
							# in the datasets folder of this project [for the dataset as input]
benchmarkfunctionfolderpath=/home/ameier/Documents/Promotion/GIT_Lab/DynOptimization/DynOpt/datasets/GECCO_2018/ # parent directory path of the benchmark functions
								   																				 # and child directory of the datasets folder
								   																				 # of this project [for the dataset as input]
outputdirectory="c1c2c3_1.49/pso_no/"	   # name of the output directory. Necessary to separate results for different algorithm settings.				
outputdirectorypath="/home/ameier/Documents/Promotion/GIT_Lab/DynOptimization/DynOpt/output/myexperiments/sphere/"		# path to output folder

# run only some experiments of all for the benchmark problem (the next four
# parameters are lists
poschgtypes=linear,sine
fitchgtypes=none
dims=2						# comma-separated integers
noises=0.0

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

# runtime
ncpus=1						# e.g. =n_repetitions 


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
-benchmarkfunctionfolderpath="$benchmarkfunctionfolderpath" \
-outputdirectory="$outputdirectory" \
-outputdirectorypath="$outputdirectorypath" \
-poschgtypes="$poschgtypes" \
-fitchgtypes="$fitchgtypes" \
-dims="$dims" \
-noises="$noises" \
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
-ncpus="$ncpus" \
