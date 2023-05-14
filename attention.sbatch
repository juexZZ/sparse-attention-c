#!/bin/bash

#SBATCH --job-name=attention
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1  
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB 
#SBATCH --time=01:00:00 
#SBATCH --verbose
#SBATCH --output=attention.out

module purge 
module load openmpi/gcc/4.0.5 

mpiexec ./main n8192d128
mpiexec ./main n1024d32