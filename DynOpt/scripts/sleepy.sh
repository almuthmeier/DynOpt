#!/bin/bash

#SBATCH --gres=gpu:0
#SBATCH --job-name="sleepy"
#SBATCH --ntasks=8
#SBATCH --mem=2G
#SBATCH --time=0-00:15
#SBATCH --output="slurm_sleepy.%j.out"
#SBATCH --error="slurm_sleepy.%j.err"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=almuth.meier@uni-oldenburg.de


python3.5 ../code/slpeepy.py 

wait