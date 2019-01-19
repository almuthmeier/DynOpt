#!/bin/bash

# like slurm.job but no .job-script

# is called by subscript.job with input parameters
# 	$1 -> learning rate
# 	$2 -> batch size
# 	$2 -> dropout rate


#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------


# benchmark problem
algorithm="dynea" 			# dynea or dynpso
repetitions=5				# number runs for each experiment
chgperiodrepetitions=1		# number runs for each change period
chgperiods=554				# number of change periods 
lenchgperiod=30				# number of generations per change period; has to be
							# set even if chgperiod==1 (is then number of 
							# generations).
							# Is required also in case ischgperiodrandom is True, 
							# because lenchgperiod*chgperiods is the number of 
							# generations. 
ischgperiodrandom="False"	# True if the change occurs at a random time point.
							# defines the benchmark function, must be located 
							# in the datasets/ folder of this project
benchmarkfunctionfolderpath="/raid/almuth/Uncertainty/Ausgaben/data_2019-01-18/" # parent directory path of the benchmark functions
								   																				 # and child directory of the datasets folder
								   																				 # of this project
lbound=0					# minimum bound of the benchmark's range							   																				 # and child directory of the datasets folder
ubound=100					# maximum bound of the benchmark's range										   																				 

# run only some experiments of all for the benchmark problem (the next four parameters are lists)
fitchgtypes=none			# fitness change type; comma-separated integers
dims=10						# dimensionality of fitness function; comma-separated integers

# PSO
c1=1.496180					# influence of particle's best solution 
c2=1.496180					# influence of swarm's best solution
c3=1.496180					# influence of prediction term
insertpred="False"			# True if predicted optimum should be inserted as individual into the population
adaptivec3="False"			# True if c3 should be changed adaptively
nparticles=200				# swarm size

# EA
mu=50						# number parents
la=100						# number offsprings
ro=2						# number parents for recombination
mean=0.0					# mutation mean
sigma=1.0					# mutation strength
trechenberg=5				# number of generations during that the number of successful mutations is counted
tau=0.5						# 0 < tau < 1, for Rechenberg

# predictor
timesteps=50				# number of previous optima used for the predictions
addnoisytraindata="False"	# True if more training data are generated by adding noise to existing data
traininterval=75			# number of change periods that must have passed before predictor is trained anew
nrequiredtraindata=128		# number of training data that is used for training
useuncs=False				# if True -> TCN with automatic learning of aleatoric and epistemic uncertainty is used;
							# ep. unc. is used as standard deviation for re-initializing the population after a change
trainmcruns=20				# only used if useuncs; number of Monte Carlo runs during training
testmcruns=5				# only used if useuncs; number of Monte Carlo runs during testing/prediction
traindropout=$3				# dropout rate for training
testdropout=$3				# only used if useuncs; dropout rate for testing/prediction
kernelsize=6	 			# kernel size for TCN
nkernels=27					# number of kernels for TCN (same in every layer)
lr=$1						# leanring rate of TCN

# ANN predictor
neuronstype=fixed20			# fixed20, dyn1.3; defines the number of neurons in the RNN prediction model (only for "rnn", not for "tfrnn")
epochs=80					# number of training epochs for the RNN prediction model
batchsize=$2				# batch size for RNN model
nlayers=2					# overall number of layers (incl. tl layers)
tlmodelpath="/raid/almuth/TransferLearning/Ausgaben/EAwithPred/models_2018-11-19/"	# path to the pre-trained transfer learning model
ntllayers=1					# number of layers in the transfer learning model
ngpus=1    					# number of GPUs to use (for RNN model) 

# runtime
ncpus=2						# e.g. =n_repetitions; number of CPUs to use (repetitions of any experiment are run parallel)  

# output paths
pathaddition="architecture/trainparams" #"firsttest"	# "stepevaluation"
#pathadditions="$pathaddition/steps""_""$timesteps"
pathadditions="$pathaddition/"

# ==================================================================================================================================================
# ==================================================================================================================================================
# ==================================================================================================================================================
# Set 1
benchmarkfunction=sphere	# sphere, griewank, rosenbrock, rastrigin, mpbnoisy, mpbrandom, mpbcorr
poschgtypes=sinefreq		# linear,sine,circle,mixture,sinefreq position change type; comma-separated integers
noises=0.00					# noise, required only for mpb-benchmarks; comma-separated floats
# ==================================================================================================================================================
# ==================================================================================================================================================
# ==================================================================================================================================================


#------------------------------------------------------------------------------
# Command 3
predictor=tcn				# no, rnn, autoregressive, tfrnn, tftlrnn, tftlrnndense

# static

algnameaddition="" 
outputdirectory="$pathadditions/$algorithm""_""$predictor""$algnameaddition/"						# name of the output directory. Necessary to separate results for different algorithm settings.				
outputdirectorypath="/raid/almuth/Uncertainty/Ausgaben/output_2019-01-18_trainparams/$benchmarkfunction/"		# path to output folder
#------------------------------------------------------------------------------

# (There must always be a space between the argument and the backslash!)

python3.5 ../code/input_parser.py -algorithm="$algorithm" \
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
-predictor="$predictor" \
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
-ncpus="$ncpus" &


wait