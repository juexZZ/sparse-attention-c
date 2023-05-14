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

### Todo list

* [ ] how to start each process (partition data according to each row's sparse index)
  * [ ] read from files: can we do parallel read? or read the whole file in process 0 and scatter (current plan)
* [ ] how to implement W x V (Need communication, W is sparse)
  * [ ] each process: part of Q, part K -> attention. which part(sparse index). part of V. Communicate the whole attention weights W.
* [ ] create sparse index list for each process (and each row)

### Data division

query, divided by the proc_id
key, dicided by the sparse pattern and proc_id, union

### 05/11/2023

- [ ] ~Juexiao: Modify the function to work with class SparsePattern and struct Id_vec~
- [X] Juexiao: Attention x Value
- [X] Yiwei: `main.cpp` line 91 double 2-d array change to 1-d, using total_count as index offset

Env: _conda deactivate_  
Compile: _make_ or _mpicxx -std=c++11 -O3 -march=native main.cpp -o main_    
Compile openmp:  _mpicxx -std=c++11 -O3 -march=native main.cpp -fopenmp -o main_  
Run: _mpirun -np <mpi_num> ./main <data_dir>_   
e.g. _mpirun -np 2 ./main n32d8_  



