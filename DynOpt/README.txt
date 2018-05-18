9.5.18

# benchmarks
## all benchmarks
- global optimum position and fitness per change
	- 'global_opt_fit_per_chgperiod'
	- 'global_opt_pos_per_chgperiod'
- original optimum position (is the same as the first entry in 'global_opt_pos_per_chgperiod' 
	- 'orig_global_opt_pos' 

- names of key: if used, "_per_chgperiod" must be at the end (because it is 
  automatically renamed to "_per_gen" 
	
## additionally in mpb benchmark
- properties of the peaks (per change)
	- 'heights'
	- 'widths'
	- 'positions'
	
	
	
	
## content of datasets folder of the project
- datasets folder must contain sub-directories each containing the benchmark
  functions for a separate conference/challenge/...


## conversion for runtime


  
# output
- output may be located at any location in the file system
- here only example output to see structure of output path

# optimization algorithms
- each one must have an optimize-Method

# input:
# lenchgperiod, and self.chgperiods have to be set even if random change time points
