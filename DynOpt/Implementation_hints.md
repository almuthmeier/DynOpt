
Implementation Hints
==================================================================================================

This file provides some useful information in case you want to extend the code by yourself.  

The code still contains some TODOs. The TODOs without tag represent improvements that will be realized in some time. TODO(exe) highlights parts in the code where the user can set parameters for the execution of experiments. TODO(dev) tags mark parts in the code where it might be necessary for developers to make adaptions if they want to integrate more functionality.  

The words "change period" and "change" are defined as follows:  
    - change period: the "time" between two changes; the first change period has index 0; should be used instead of "change"  
    - change: means that the fitness function has changed; the first change has index 0; "change period" should be used instead, otherwise the code could become confusing if both words are used.  



## Benchmark files

The generated data set files contain all information to compute the fitness for any point in the solution space at a specific generation. The data set folder should contain sub-directories each containing the benchmark functions for a separate conference, challenge, etc. The folder for one benchmark function can contain different .npz-files. Each such file represents a specific instance of the benchmark, e.g., the benchmark can be instantiated with different dimensionalities.

- datasets/  
    - conference/  
        - benchmark1/  
            - instanceA.npz  
            - instanceB.npz  
        - benchmark2/  
        ...  

The properties of an instance can be seen in the file name, e.g.:  
    - mpbnoisy_d-50_chgperiods-10000_veclen-0.6_peaks-10_noise-0.0_2018-05-09_11:20.npz  
    - rosenbrock_d-50_chgperiods-10000_pch-sine_fch-none_2018-05-09_11:13.npz  
Since information will be read out from the file name its structure must be like : `<benchmark name>`\_`<property1>`-`<value of property1>`\_`<property2>`-`<value of property2>`\_...

  
  
### Content of all benchmark files

Every .npz file contains the following entries:  
    - 'global_opt_fit_per_chgperiod' (global optimum fitness per change; 1d numpy array: for each change period the optimum fitness)  
	- 'global_opt_pos_per_chgperiod' (global optimum position per change; 2d numpy array: for each change period the position corresponding to the optimum fitness)  
	- 'orig_global_opt_pos' (original (unmoved) optimum position:
		- in case of Sphere, Rastrigin, Rosenbrock,... as base functions it is the position of the global optimum in the unmoved base function (there fore not necessarily equal to first entry in 'global_opt_pos_per_chgperiod' (e.g. for position change type "mixture" not equal)
		- for the MPB variants it is the same as the first entry in 'global_opt_pos_per_chgperiod'  
The first two listed keys must have "_per_chgperiod" at the end because they are automatically renamed to "_per_gen" in comparison.convert_data_to_per_generation().  

Note that it is important that the properties are stored "per change" and not "per generation" as they are converted to "per generation" in comparison.py during runtime.


### Additional content of mpb benchmark files

For the mpb benchmarks it is necessary to have the heights, widths, and positions of the peaks (per change):  
	- 'heights'  
	- 'widths'  
	- 'positions'  



## Output of experiments
For each run of an experiment one .npz-file is stored that contains information about the optimization process. It contains the following eight elements that are necessary to compute the metric values afterwards:  

'best_found_fit_per_gen' (fitness of best found individual for each generation (1d numpy array))  
'best_found_pos_per_gen' (best found individual for each generation (2d numpy array))  
'best_found_fit_per_chgperiod' (fitness of found optima (one for each change period))  
'best_found_pos_per_chgperiod' (position of found optima (one for each change period))  
'pred_opt_fit_per_chgperiod' (fitness of predicted optima (one for each change period))  
'pred_opt_pos_per_chgperiod' (position of predicted optima (one for each change period))  
'detected_chgperiods_for_gens' (for each detected change the corresponding generation numbers)  
'real_chgperiods_for_gens' (1d numpy array containing for each generation the change period number it belongs to)  

The output .npz-files may be located at any location in the file system.


## Optimization algorithms

A new optimization algorithm can easily be implemented. It needs an optimize() method that conducts one complete optimization run. In addition, most of the variables from the __init__ method have to be taken over e.g. to define the benchmark function or to set all necessary variables that will be read for the output.


## Metric calculator

metric_calculator.py generates a result file containing the metric values per run. Note: The BOG is averaged over generations leading to one value per run. In its original definition the BOG is averaged over the runs.


