
Dynamic Optimization
==================================================================================================

This software package provides python code to reproduce the experiments we conducted for our ICANN 2019 paper [3]. In order to get the code for our GECCO 2018 and Evo* 2018 papers [1, 2], reset to the respective tag ("GECCO18-release") in the GIT history. Due to some reconstructions (e.g., changes of the seeds), the exact values from the papers might not be reproducible.

[1] A. Meier, O. Kramer: Recurrent Neural Network-Predictions for PSO in Dynamic Optimization, EvoApplications 2018.  
[2] A. Meier, O. Kramer: Prediction with Recurrent Neural Networks in Evolutionary Dynamic Optimization, GECCO 2018.  
[3] A. Meier, O. Kramer: Predictive Uncertainty Estimation with Temporal Convolutional Networks for Dynamic Evolutionary Optimization, ICANN 2019.

## Requirements & Installation
In order to run this python project you need a python installation and Keras with TensorFlow to run the recurrent neural network prediction models. Use the requirements.txt to ensure you have all necessary python packages, but not all packages listed in the requirements.txt are obligatory.

The code has been tested with:  
    - Python 3.5.2  
    - Keras 2.1.1  
    - TensorFlow 1.5.0  
    - Cuda 9.0.176  
    - cuDNN 7.0.5  
    - Ubuntu 16.04  
    
## Directory Structure
Since in GIT it is not possible to commit empty directories you have to create some directories in order to get the following structure:  

- DynOpt/  
    - code/  
        - (everyting is already there)  
    - datasets/  
        - [eventname for that the benchmark files are required] (sub-directories are generated   automatically)  
    - output/  
        - [eventname for that the benchmark files are required] (same as under datasets/)  
    - scripts/  
        - (everything is already there)  

## Usage

1. Generation of benchmark files
The directory datasets/EvoStar_2018/ contains some test data sets. To run a more comprehensive experimental study more benchmarks can be created by running the python scripts in the code/benchmarks/ folder. The desired properties of the fitness functions have to be specified there.
    
2. Execution of experiments    
To run the experiments, execute either the run_local.sh (in scripts/) or input_parser.py (in code/) after defining, e.g., which of the generated data sets should be used. For each run of an experiment the output of the algorithm is stored in the output/ directory in the corresponding array/ subdirectory.

3. Computation of metrics  
The quality metrics are computed by running the code/metrics/metric_calculator.py. Therefore, the files generated during the execution of the experiments are required. The metrics for all runs, all algorithms and all experiments are stored in one file (metric_db.csv) in the output directory.

4. Conduction of statistical tests  
In order to conduct the statistical tests, run code/metrics/stattest_calculator.py. It takes metric_db.csv as input and outputs for each possible algorithm pair one file in the stattest subdirectory.

Further explanations about the usage of the modules can be found in the [implementation hints](Implementation_hints.md) and in the code itself.

## Dynamic Sine Benchmark

In the ICANN 2019 paper we propose a new benchmark generator for dynamic optimization problems. It is implemented in benchmarks/sine_generator.py. In order to create a data set:

* specify parameters in start_generation() and generate_sine_fcts_for_one_dimension()
* run the main function
* save the data at the desired location

## Updates
### November 2018
- returnseq=True only possible with tftlrnn and tfrnn and tftlrnndense

### March 2019
- added code for ICANN 2019 paper

### May 2019
- corrected ICANN 2019 code (parameters for normal deviation)



- "truepred" ist noch nicht in Re-initialisierungsstrategien eingebaut (25.7.19)
	-> compute_noisy_opt_positions() in dynea.py
- epist_unc_per_chgperiod und kal_variance_per_chgperiod durch pred_unc_per_chgperiod ersetzt; wird also von allen Vorhersagemethoden, die eine Unsicherheitsschätzung haben, verwendet. aleat_unc_per_chgperiod ist weiterhin nur für TCN 