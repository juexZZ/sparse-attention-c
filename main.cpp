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
    int N = 1024;
    int d = 128;
    int c = 16;
    string query_file = "query.txt";
    string key_file = "key.txt";
    string value_file = "value.txt";
    int num_procs, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // read data and distribute to other processes
    // each process hold: part_query, part_key, part_value
    double* part_query;
    double* part_key;
    double* part_value;

    vector<int> row_ids; // the rows this process need to work on for attention
    vector<int> total_col_ids; // the column idx this process needs to process, in total
    vector<vector<int>> col_ids_row(row_ids.size(), vector<int>(0,0)); // each row's column idx (for sparse attention)    

    int pattern=0; //Two patterns, 0 is strided, 1 is fixed.
    get_row_share(my_rank,num_procs, N, row_ids);
    get_column_share(my_rank, N, d, row_ids, total_col_ids, col_ids_row, pattern);
    if (my_rank==0){
        double* query = new double[N*d];
        for(int i=0; i<N; i++){
            query[i] = new double[d];
        }
        read_data(query, N, d, query_file);
        // todo: distribute to other processes
        // MPI, send data to proc0ess

        // then free query
        delete[] query;

        // read key
        double* key = new double[N*d];
        read_data(key, N, d, key_file);
        // todo: distribute to other processes

        delete[] key;
        // read value
        double* value = new double[N*d];
        read_data(value, N, d, value_file);
        // todo: distribute to other processes

        delete[] value;
    }
    // sparse attention
    double** attn_w = new double*[row_ids.size()];
    for(int r=0; r<row_ids.size(); r++){
        attn_w[r] = new double[col_ids_row[r].size()];
        row_sparse_attention(part_query[r], part_key, attn_w[r], col_ids_row[r].size(), d);
    }
    // communicate attention weights
    
    // attention times value

    MPI_Finalize();

}