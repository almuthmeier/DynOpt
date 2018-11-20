
# This script executes the specified experiments.
#
# Run this script from command line with: sh slurm_vegas.sh


#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------

# benchmark problem
algorithm="dynea" 			# dynea or dynpso
repetitions=5				# number runs for each experiment
chgperiods=500				# number of change periods 
lenchgperiod=20				# number of generations per change period; has to be
							# set even if chgperiod==1 (is then number of 
							# generations).
							# Is required also in case ischgperiodrandom is True, 
							# because lenchgperiod*chgperiods is the number of 
							# generations. 
ischgperiodrandom="False"	# True if the change occurs at a random time point.
							# defines the benchmark function, must be located 
							# in the datasets/ folder of this project
benchmarkfunctionfolderpath="/home/almuth/Documents/Promotion/Ausgaben/TransferLearning/EAwithPred/data_2018-11-19/" # parent directory path of the benchmark functions
								   																				 # and child directory of the datasets folder
								   																				 # of this project 

# run only some experiments of all for the benchmark problem (the next four
# parameters are lists)
fitchgtypes=none			# fitness change type; comma-separated integers
dims=1,5,10,50				# dimensionality of fitness function; comma-separated integers

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
timesteps=100				# number of previous optima used for the predictions

# ANN predictor
neuronstype=fixed20			# fixed20, dyn1.3; defines the number of neurons in the RNN prediction model (only for "rnn", not for "tfrnn")
epochs=5					# number of training epochs for the RNN prediction model
batchsize=128					# batch size for RNN model
nlayers=2					# overall number of layers (incl. tl layers)
tlmodelpath=""				# path to the pre-trained transfer learning model
ntllayers=1					# number of layers in the transfer learning model
ngpus=1    					# number of GPUs to use (for RNN model) 

# runtime
ncpus=2						# e.g. =n_repetitions; number of CPUs to use (repetitions of any experiment are run parallel)  



# ==================================================================================================================================================
# ==================================================================================================================================================
# ==================================================================================================================================================
# Set 1
benchmarkfunction=mpbcorr	# sphere, griewank, rosenbrock, rastrigin, mpbnoisy, mpbrandom, mpbcorr
poschgtypes=none			# linear,sine,circle,mixture position change type; comma-separated integers
noises=0.0					# noise, required only for mpb-benchmarks; comma-separated floats
# ==================================================================================================================================================
# ==================================================================================================================================================
# ==================================================================================================================================================


#------------------------------------------------------------------------------
# Command 2
predictor=autoregressive	# no, rnn, autoregressive, tfrnn, tftlrnn

# static
outputdirectory="stepevaluation/steps""_""$timesteps/$algorithm""_""$predictor/" 									# name of the output directory. Necessary to separate results for different algorithm settings.				
outputdirectorypath="/home/almuth/Documents/Promotion/Ausgaben/TransferLearning/EAwithPred/output_2018-11-20/$benchmarkfunction/"		# path to output folder
#------------------------------------------------------------------------------

python3.5 ../code/input_parser.py -algorithm="$algorithm" \
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
-nlayers="$nlayers" \
-tlmodelpath="$tlmodelpath" \
-ntllayers="$ntllayers" \
-ngpus="$ngpus" \
-ncpus="$ncpus" &

#------------------------------------------------------------------------------

# ==================================================================================================================================================
# ==================================================================================================================================================
# ==================================================================================================================================================
# Set 2
benchmarkfunction=sphere	# sphere, griewank, rosenbrock, rastrigin, mpbnoisy, mpbrandom, mpbcorr
poschgtypes=mixture			# linear,sine,circle,mixture position change type; comma-separated integers
noises=0.0					# noise, required only for mpb-benchmarks; comma-separated floats
# ==================================================================================================================================================
# ==================================================================================================================================================
# ==================================================================================================================================================



#------------------------------------------------------------------------------
# Command 2
predictor=autoregressive	# no, rnn, autoregressive, tfrnn, tftlrnn

# static
outputdirectory="stepevaluation/steps""_""$timesteps/$algorithm""_""$predictor/" 									# name of the output directory. Necessary to separate results for different algorithm settings.				
outputdirectorypath="/home/almuth/Documents/Promotion/Ausgaben/TransferLearning/EAwithPred/output_2018-11-20/$benchmarkfunction/"		# path to output folder
#------------------------------------------------------------------------------

python3.5 ../code/input_parser.py -algorithm="$algorithm" \
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
-nlayers="$nlayers" \
-tlmodelpath="$tlmodelpath" \
-ntllayers="$ntllayers" \
-ngpus="$ngpus" \
-ncpus="$ncpus" &


# ==================================================================================================================================================
# ==================================================================================================================================================
# ==================================================================================================================================================
# Set 3
benchmarkfunction=rastrigin	# sphere, griewank, rosenbrock, rastrigin, mpbnoisy, mpbrandom, mpbcorr
poschgtypes=mixture			# linear,sine,circle,mixture position change type; comma-separated integers
noises=0.0					# noise, required only for mpb-benchmarks; comma-separated floats
# ==================================================================================================================================================
# ==================================================================================================================================================
# ==================================================================================================================================================


#------------------------------------------------------------------------------
# Command 2
predictor=autoregressive	# no, rnn, autoregressive, tfrnn, tftlrnn

# static
outputdirectory="stepevaluation/steps""_""$timesteps/$algorithm""_""$predictor/" 									# name of the output directory. Necessary to separate results for different algorithm settings.				
outputdirectorypath="/home/almuth/Documents/Promotion/Ausgaben/TransferLearning/EAwithPred/output_2018-11-20/$benchmarkfunction/"		# path to output folder
#------------------------------------------------------------------------------

python3.5 ../code/input_parser.py -algorithm="$algorithm" \
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
-nlayers="$nlayers" \
-tlmodelpath="$tlmodelpath" \
-ntllayers="$ntllayers" \
-ngpus="$ngpus" \
-ncpus="$ncpus" &

#------------------------------------------------------------------------------

# ==================================================================================================================================================
# ==================================================================================================================================================
# ==================================================================================================================================================
# Set 4
benchmarkfunction=griewank	# sphere, griewank, rosenbrock, rastrigin, mpbnoisy, mpbrandom, mpbcorr
poschgtypes=mixture			# linear,sine,circle,mixture position change type; comma-separated integers
noises=0.0					# noise, required only for mpb-benchmarks; comma-separated floats
# ==================================================================================================================================================
# ==================================================================================================================================================
# ==================================================================================================================================================

#------------------------------------------------------------------------------
# Command 2
predictor=autoregressive	# no, rnn, autoregressive, tfrnn, tftlrnn

# static
outputdirectory="stepevaluation/steps""_""$timesteps/$algorithm""_""$predictor/" 									# name of the output directory. Necessary to separate results for different algorithm settings.				
outputdirectorypath="/home/almuth/Documents/Promotion/Ausgaben/TransferLearning/EAwithPred/output_2018-11-20/$benchmarkfunction/"		# path to output folder
#------------------------------------------------------------------------------

python3.5 ../code/input_parser.py -algorithm="$algorithm" \
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
-nlayers="$nlayers" \
-tlmodelpath="$tlmodelpath" \
-ntllayers="$ntllayers" \
-ngpus="$ngpus" \
-ncpus="$ncpus" &

#------------------------------------------------------------------------------


wait