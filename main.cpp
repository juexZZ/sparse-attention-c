// main cpp for sparse attention
// author: Juexiao Zhang, Yiwei Shao
// May 2023
#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <mpi.h>
#include <omp.h>
#include "utils.h"

using namespace std;

int main(int argc, char* argv[]){
    // read query, key and value from seperate files

    //Two communication patterns, 0 prepare all the data needed before computing, 1 compute and communicate at the same time.
    int comm_pat=0; 

    int N = 1024;
    int d = 128;
    int context_l = 16; // paper l
    int fixed_c = 0; // paper c
    string query_file = "query.txt";
    string key_file = "key.txt";
    string value_file = "value.txt";
    int num_procs, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // read data and distribute to other processes
    // each process hold: part_query_first, part_key_first, part_value_first, part_query_last, part_key_last, part_value_last
    // to keep thread load balanced

    SparsePattern pattern_first;
    SparsePattern pattern_last;

    int pattern=0; //Two patterns, 0 is strided, 1 is fixed.
    get_row_share(my_rank,num_procs, N, row_ids_first, row_ids_last);
    get_column_share(my_rank, N, d, pattern_first, pattern, context_l, fixed_c);
    get_column_share(my_rank, N, d, pattern_last, pattern, context_l, fixed_c);

    int row_size=pattern_first.start_row_id-pattern_first.end_row_id;
    double** part_query_first[row_size][d];
    double** part_key_first[row_size][d];
    double** part_value_first[row_size][d];

    double** part_query_last[row_size][d];
    double** part_key_last[row_size][d];
    double** part_value_last[row_size][d];

    int* sendscounts= new int[num_procs];
    int* displs_first = new int[num_procs];
    int* displs_last = new int[num_procs];
    for (size_t i = 0; i < num_procs; i++)
    {
        int start_id=N/num_procs/2*i;
        int end_id=N/num_procs/2*(i+1);
        displs_first[i]=start_id*d;
        sendscounts[i]=(end_id-start_id)*d;
        displs_last[i]=(N-end_id)*d;
    }

    if (my_rank==0){    
        double query[N][d];
        read_data(query, N, d, query_file);
        MPI_Scatterv(query, sendscounts, displs_first, MPI_DOUBLE, part_query_first, sendscounts[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(query, sendscounts, displs_last, MPI_DOUBLE, part_query_last, sendscounts[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // then free query
        delete[] query;

        // read key
        double key[N][d];
        read_data(key, N, d, key_file);
        // todo: distribute to other processes
        MPI_Scatterv(key, sendscounts, displs_first, MPI_DOUBLE, part_key_first, sendscounts[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(key, sendscounts, displs_last, MPI_DOUBLE, part_key_last, sendscounts[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        delete[] key;
        // read value
        double value[N][d];
        read_data(value, N, d, value_file);
        // todo: distribute to other processes
        MPI_Scatterv(value, sendscounts, displs_first, MPI_DOUBLE, part_value_first, sendscounts[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(value, sendscounts, displs_last, MPI_DOUBLE, part_value_last, sendscounts[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        delete[] value;
    }
    // sparse attention
    double** attn_w_first = new double*[row_ids_first.size()];
    double** attn_w_last = new double*[row_ids_last.size()];
    if (comm_pat==0)
    {
        attn_w_first[r] = new double[col_ids_row_first[r].size()];
        delete[] full_keys;
    }
    
    for(int r=0; r<row_ids_first.size(); r++){
        attn_w_first[r] = new double[col_ids_row_first[r].size()];
        row_sparse_attention(part_query_first[r], part_key_first, attn_w_first[r], col_ids_row_last[r].size(), d);
    }
    for (int r = 0; r < row_ids_last.size(); r++)
    {
        attn_w_last[r] = new double[col_ids_row_last[r].size()];
        row_sparse_attention(part_query_last[r], part_key_last, attn_w_last[r], col_ids_row_last[r].size(), d);
    }
    // communicate attention weights

    // attention times value

    // free attn_w
    for(int r=0; r<row_ids.size(); r++){
        delete[] attn_w[r];
    }
    delete[] attn_w;
    // free the rest
    delete[] part_query_first;
    delete[] part_key_first;
    delete[] part_value_first;
    delete[] part_query_last;
    delete[] part_key_last;
    delete[] part_value_last;
    
    delete[] sendscounts;
    delete[] displs_first;
    delete[] displs_last;

    MPI_Finalize();

}