# sparse-attention-c

NYU HPC Spring 23 Course Project

### Todo list

* [ ] how to start each process (partition data according to each row's sparse index)
  * [ ] read from files: can we do parallel read? or read the whole file in process 0 and scatter (current plan)
* [ ] how to implement W x V (Need communication, W is sparse)
  * [ ] each process: part of Q, part K -> attention. which part(sparse index). part of V. Communicate the whole attention weights W.
* [ ] create sparse index list for each process (and each row)

### Data division
query, divided by the proc_id
key, dicided by the sparse pattern and proc_id, union