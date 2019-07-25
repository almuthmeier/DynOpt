#!/bin/bash

# This script executes the specified experiments.
#
# Run this script from command line with: ./run_local.sh 
#
# In order to run this script from eclipse:
# https://askubuntu.com/questions/514247/running-a-bash-file-in-eclipse (11.5.18)
#	- make script runnable: chmod +x run_local.sh
#	- in eclipse: menu Run -> External Tools -> External Tool Configurations
#		- double-click on "Program"
#		- Location: search for script
#		- Working Directory: directory that contains the script

#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------

# benchmark problem
algorithm="dyncma" 			# dynea or dynpso
repetitions=1				# number runs for each experiment
chgperiodrepetitions=1		# number runs for each change period
chgperiods=50				# number of change periods 
lenchgperiod=10				# number of generations per change period; has to be
							# set even if chgperiod==1 (is then number of 
							# generations).
							# Is required also in case ischgperiodrandom is True, 
							# because lenchgperiod*chgperiods is the number of 
							# generations. 
ischgperiodrandom="False"	# True if the change occurs at a random time point.
benchmarkfunction=sphere	# sphere, rosenbrock, rastrigin, mpbnoisy, mpbrandom, mpbcorr
							# defines the benchmark function, must be located 
							# in the datasets/ folder of this project
benchmarkfunctionfolderpath="/home/ameier/Documents/Promotion/Ausgaben/Uncertainty/Ausgaben/data_2019-01-19_final/" # parent directory path of the benchmark functions
								   																				 # and child directory of the datasets folder
								   																				 # of this project
lbound=0					# minimum bound of the benchmark's range							   																				 
ubound=100					# maximum bound of the benchmark's range										   																				 

# run only some experiments of all for the benchmark problem (the next four parameters are lists)
poschgtypes=sinefreq		# linear,sine,circle,mixture,sinefreq position change type; comma-separated strings
fitchgtypes=none			# fitness change type; comma-separated integers
dims=2						# dimensionality of fitness function; comma-separated integers
noises=0.0					# noise, required only for mpb-benchmarks; comma-separated floats
							# attention: for correct log-file name it is assumed that noises contain only a single value!

# PSO
c1=1.496180					# influence of particle's best solution 
c2=1.496180					# influence of swarm's best solution
c3=1.496180					# influence of prediction term
insertpred="False"			# True if predicted optimum should be inserted as individual into the population
adaptivec3="False"			# True if c3 should be changed adaptively
nparticles=200				# swarm size

# EA
mu=5						# number parents
la=10						# number offsprings
ro=2						# number parents for recombination
mean=0.0					# mutation mean
sigma=1.0					# mutation strength
trechenberg=5				# number of generations during that the number of successful mutations is counted
tau=0.5						# 0 < tau < 1, for Rechenberg
reinitializationmode="no-RND" # mode for re-initialization of the population: "no-RND" "no-VAR" "no-PRE" "pred-RND" "pred-UNC" "pred-DEV" "pred-KAL"
sigmafactors=0.01,0.1		# list of floats, factors for the sigma environment for random population re-initialization

# CMA-ES
cmavariant="predcma_internal" # variant how CMA-ES includes prediction or path estimation for dynamic: "resetcma" "predcma_internal" "predcma_external"
predvariant="h"				# variant how to calculate sig/m after a change: "simplest", "a", "b", "c", "d", "g" ,"branke", "f", "ha", "hb", "hd", "hawom", "hbwom", "hdwom"

# predictor
predictor=truepred			# no, rnn, autoregressive, tfrnn, tftlrnn, tftlrnndense, kalman, truepred
							# prediciton model to predict the new optimum	
trueprednoise=0.1			# noise of prediction with "truepred" (known optimum disturbed with noise that is normally distributed with standarddeviation "trueprednoise") 												
timesteps=4				  	# number of previous optima used for the predictions
addnoisytraindata="False"	# True if more training data are generated by adding noise to existing data
traininterval=5				# number of change periods that must have passed before predictor is trained anew
nrequiredtraindata=10		# number of training data that is used for training
useuncs=False				# if True -> TCN with automatic learning of aleatoric and epistemic uncertainty is used;
							# ep. unc. is used as standard deviation for re-initializing the population after a change
							# only possible for prediction modes "kalman" and "tcn" 
trainmcruns=20				# only used if useuncs; number of Monte Carlo runs during training
testmcruns=5				# only used if useuncs; number of Monte Carlo runs during testing/prediction
traindropout=0.1			# dropout rate for training
testdropout=0.1				# only used if useuncs; dropout rate for testing/prediction
kernelsize=3				# kernel size for TCN
nkernels=16					# number of kernels for TCN (same in every layer)
lr=0.001					# leanring rate of TCN

# ANN predictor
neuronstype=fixed20			# fixed20, dyn1.3; defines the number of neurons in the RNN prediction model (only for "rnn", not for "tfrnn")
epochs=5					# number of training epochs for the RNN prediction model
batchsize=32				# batch size for RNN model
nlayers=2					# overall number of layers (incl. tl layers)
tlmodelpath="/home/ameier/Documents/Promotion/Ausgaben/TransferLearning/TrainTLNet/Testmodell/"	# path to the pre-trained transfer learning model
ntllayers=1					# number of layers in the transfer learning model
ngpus=1    					# number of GPUs to use (for RNN model) 

# runtime
ncpus=2						# e.g. =n_repetitions; number of CPUs to use (repetitions of any experiment are run parallel)  

# output paths: TODO adapt if necessary
pathaddition="scriptout" #"firsttest"	# "stepevaluation"
pathadditions="$pathaddition"
pathadditions=""

#algnameaddition="_""test"
algnameaddition="" 
outputdirectory="$pathadditions/$algorithm""_""$predictor""$algnameaddition/"						# name of the output directory. Necessary to separate results for different algorithm settings.				
outputdirectorypath="/home/ameier/Documents/Promotion/GIT_Lab/DynOptimization/DynOpt/output/EvoStar_2020/$benchmarkfunction/"		# path to output
																																	# folder

#------------------------------------------------------------------------------
# Command
#------------------------------------------------------------------------------

# (There must always be a space between the argument and the backslash!)

~/.virtualenvs/promotion/prototype/bin/python3.5 ../code/input_parser.py -algorithm="$algorithm" \
-repetitions="$repetitions" \
-chgperiodrepetitions="$chgperiodrepetitions" \
-chgperiods="$chgperiods" \
-lenchgperiod="$lenchgperiod" \
-ischgperiodrandom="$ischgperiodrandom" \
-benchmarkfunction="$benchmarkfunction" \
-benchmarkfunctionfolderpath="$benchmarkfunctionfolderpath" \
-lbound="$lbound" \
-ubound="$ubound" \
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
-reinitializationmode="$reinitializationmode" \
-sigmafactors="$sigmafactors" \
-cmavariant="$cmavariant" \
-predvariant="$predvariant" \
-predictor="$predictor" \
-trueprednoise="$trueprednoise" \
-timesteps="$timesteps" \
-addnoisytraindata="$addnoisytraindata" \
-traininterval="$traininterval" \
-nrequiredtraindata="$nrequiredtraindata" \
-useuncs="$useuncs" \
-trainmcruns="$trainmcruns" \
-testmcruns="$testmcruns" \
-traindropout="$traindropout" \
-testdropout="$testdropout" \
-kernelsize="$kernelsize" \
-nkernels="$nkernels" \
-lr="$lr" \
-neuronstype="$neuronstype" \
-epochs="$epochs" \
-batchsize="$batchsize" \
-nlayers="$nlayers" \
-tlmodelpath="$tlmodelpath" \
-ntllayers="$ntllayers" \
-ngpus="$ngpus" \
-ncpus="$ncpus"