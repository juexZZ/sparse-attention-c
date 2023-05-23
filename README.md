# sparse-attention-c

NYU HPC Spring 23 Course Project

### Generate data

python and numpy is required to `data_gen.py`. For example:

```python
python data_gen.py 100 10 random
python data_gen.py 100 10 ordered
```
data will be stored as `query.txt`, `key.txt` and `value.txt` respectively.

### Module
module load gcc/10.2.0  
module load openmpi/gcc/4.0.5 

## Experiment Setup
Compile: `make` or `mpicxx -std=c++11 -O3 -march=native main.cpp -o main`    
Compile openmp: `mpicxx -std=c++11 -O3 -march=native main.cpp -fopenmp -o main`  
Run: `mpirun -np <mpi_num> ./main <data_dir>`   
e.g. `mpirun -np 2 ./main n32d8`  
You can use [attention.sbatch](attention.sbatch) to submit single task or use `bash submit.sh` to submit tasks of different nodes.

## Result
You can see our results in [result](result). [result.ipynb](result/result.ipynb) is used to parse the result log file and plot.
## Test Experiment
Codes in [directory](omp_test) is used to test the OpenMP on Greene. [omp_hello](omp_test/omp_hello.cpp) is used to test if a pure OpenMP program can be executed correctly on Greene, and [mpi_hello](omp_test/mpi_hello.cpp) is used to test if a hybrid MPI and OpenMP program can be executed correctly. [mpi_submit.sh](omp_test/mpi_submit.sh) and [omp_submit.sh](omp_test/omp_submit.sh) are corresponding shell scripts to submit tasks. Pure OpemMP programs run successfully on Greene but unfortunately MPI and OpenMP hybrid program cannot run OpenMP parallel. 



