for l in 8192  
do
    for n in 1 
    do
        for t in 1 2 4 8 16
        do
cat <<EOF >n${n}_t${t}_l${l}.sbatch
#!/bin/bash

#SBATCH --job-name=attention
#SBATCH --nodes=${n}
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=${t}
#SBATCH -c ${t}
#SBATCH --mem=6GB 
#SBATCH --time=01:00:00 
#SBATCH --verbose
#SBATCH --output=result_thread/n${n}_t${t}.out

module purge 
module load gcc/10.2.0
module load openmpi/gcc/4.0.5 
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OMP_THREAD_LIMIT=\$SLURM_CPUS_PER_TASK
mpiexec ./main n${l}d1024 ${t}
EOF
    sbatch n${n}_t${t}_l${l}.sbatch
    rm n${n}_t${t}_l${l}.sbatch
        done
    done
done
