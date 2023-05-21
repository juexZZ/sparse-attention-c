for t in 1 2 4 8 16
do
cat <<EOF >t${t}.sbatch
#!/bin/bash

#SBATCH --job-name=attention
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=${t}
#SBATCH --mem=2GB 
#SBATCH --time=01:00:00 
#SBATCH --verbose
#SBATCH --output=result_omp/t${t}.out

module purge 
module load gcc/10.2.0
module load openmpi/gcc/4.0.5 
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OMP_THREAD_LIMIT=\$SLURM_CPUS_PER_TASK
./hello 
EOF
    sbatch t${t}.sbatch
    rm t${t}.sbatch
done
