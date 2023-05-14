#!/bin/bash

#SBATCH --job-name=attention
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=4
#SBATCH --mem=2GB 
#SBATCH --time=01:00:00 
#SBATCH --verbose
#SBATCH --output=n4_t4.out

module purge 
module load gcc/10.2.0
module load openmpi/gcc/4.0.5 
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_THREAD_LIMIT=$SLURM_CPUS_PER_TASK
mpiexec ./main n8192d512
