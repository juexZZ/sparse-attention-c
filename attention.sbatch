#!/bin/bash

#SBATCH --job-name=attention
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1  
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB 
#SBATCH --time=01:00:00 
#SBATCH --verbose
#SBATCH --output=attention_4n_2t.out

module purge 
module load openmpi/gcc/4.0.5 

mpiexec ./main n8192d512
mpiexec ./main n8192d128
mpiexec ./main n1024d32

