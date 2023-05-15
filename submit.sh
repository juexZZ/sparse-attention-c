for l in 2048 4096 8192 16384 32768 
do
    for n in 1 2 4 8 16
    do
        for t in 1
        do
cat <<EOF >n${n}_t${t}_l${l}.sbatch
#!/bin/bash

#SBATCH --job-name=attention
#SBATCH --nodes=${n}
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=${t}
#SBATCH --mem=6GB 
#SBATCH --time=01:00:00 
#SBATCH --verbose
#SBATCH --output=result/n${l}d1024/n${n}_t${t}.out

module purge 
module load gcc/10.2.0
module load openmpi/gcc/4.0.5 
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OMP_THREAD_LIMIT=\$SLURM_CPUS_PER_TASK
mpiexec ./main n${l}d1024
EOF
    sbatch n${n}_t${t}_l${l}.sbatch
    rm n${n}_t${t}_l${l}.sbatch
        done
    done
done
