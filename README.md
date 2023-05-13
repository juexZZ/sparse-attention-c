# sparse-attention-c

NYU HPC Spring 23 Course Project

### Generate data

python and numpy is required to `data_gen.py`. For example:

```python
python data_gen.py 100 10 random
python data_gen.py 100 10 ordered
```

data will be stored as `query.txt`, `key.txt` and `value.txt` respectively.

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

Yiwei: ~~scatter key and value according to cols_id to different processes~~
Might need to combine First SparsePattern and Last SparsePattern to avoid repeat

- [ ] ~Juexiao: Modify the function to work with class SparsePattern and struct Id_vec~
- [X] Juexiao: Attention x Value
- [ ] Yiwei: `main.cpp` line 91 double 2-d array change to 1-d, using total_count as index offset
