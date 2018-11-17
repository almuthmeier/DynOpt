#!/bin/bash

# This script executes the specified experiments.
#
# Run this script from command line with: ./run.sh 
#
# In order to run this script from eclipse:
# https://askubuntu.com/questions/514247/running-a-bash-file-in-eclipse (11.5.18)
#	- make script runnable: chmod +x run.sh
#	- in eclipse: menu Run -> External Tools -> External Tool Configurations
#		- double-click on "Program"
#		- Location: search for script
#		- Working Directory: directory that contains the script

#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------

# benchmark problem
algorithm="dynea" 			# dynea or dynpso
repetitions=1				# number runs for each experiment
chgperiods=10				# number of change periods 
lenchgperiod=20				# number of generations per change period; has to be
							# set even if chgperiod==1 (is then number of 
							# generations).
							# Is required also in case ischgperiodrandom is True, 
							# because lenchgperiod*chgperiods is the number of 
							# generations. 
ischgperiodrandom="False"	# True if the change occurs at a random time point.
benchmarkfunction=rastrigin	# sphere, rosenbrock, rastrigin, mpbnoisy, mpbrandom
							# defines the benchmark function, must be located 
							# in the datasets/ folder of this project
benchmarkfunctionfolderpath=/home/ameier/Documents/Promotion/GIT_Lab/DynOptimization/DynOpt/datasets/GECCO_2019/ # parent directory path of the benchmark functions
								   																				 # and child directory of the datasets folder
								   																				 # of this project 

# run only some experiments of all for the benchmark problem (the next four
# parameters are lists)
poschgtypes=linear,sine,circle	# position change type; comma-separated integers
fitchgtypes=none				# fitness change type; comma-separated integers
dims=2							# dimensionality of fitness function; comma-separated integers
noises=0.0						# noise, required only for mpb-benchmarks; comma-separated floats

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

# predictor
predictor=no				# no, rnn, autoregressive, tltfrnn, tfrnn
							# prediciton model to predict the new optimum
timesteps=7					# number of previous optima used for the predictions

# ANN predictor
neuronstype=fixed20			# fixed20, dyn1.3; defines the number of neurons in the RNN prediction model
epochs=30					# number of training epochs for the RNN prediction model
batchsize=1					# batch size for RNN model
ngpus=1    					# number of GPUs to use (for RNN model) 

# runtime
ncpus=2						# e.g. =n_repetitions; number of CPUs to use (repetitions of any experiment are run parallel)  

# output: TODO adapt if necessary
outputdirectory="c1c2c3_1.49/$algorithm""_""$predictor/" 				# name of the output directory. Necessary to separate results
																		# for different algorithm settings.				
outputdirectorypath="/home/ameier/Documents/Promotion/GIT_Lab/DynOptimization/DynOpt/output/GECCO_2019/$benchmarkfunction/"		# path to output
																																	# folder

#------------------------------------------------------------------------------
# Command
#------------------------------------------------------------------------------

# (There must always be a space between the argument and the backslash!)

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
-ncpus="$ncpus"