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
	
	
## file name
- mpbrand_d-50_chgperiods-10000_veclen-0.6_peaks-10_noise-none_2018-05-09_11:19.npz
- rosenbrock_d-50_chgperiods-10000_pch-sine_fch-none_2018-05-09_11:13.npz
- gdbg_d-..._chgperiods-..._f-..._c-..._2018-05-..._11:04.npz
	
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


## metric_calculator
- the result file contains values for all runs. If average values/... are required that has to be done separately (TODO)
