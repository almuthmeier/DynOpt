#!/bin/bash

kernelsizes=(2,3,4,5)
for k in "${kernelsizes[@]}"
do
	sbatch subscript.job $k &
done

wait