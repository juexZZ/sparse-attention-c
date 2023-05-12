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
    get_row_share(my_rank,num_procs, N, pattern_first, pattern_last);
    get_column_share(my_rank, N, d, pattern_first, pattern, context_l, fixed_c);
    get_column_share(my_rank, N, d, pattern_last, pattern, context_l, fixed_c);

    // For query communication
    double query[N][d];
    double key[N][d]; // Here key is transpose
    double value[N][d];
    if (my_rank==0){    
       
        read_data(query, N, d, query_file);
        // then free query
        // read key
       
        read_data(key, N, d, key_file);
        // todo: distribute to other processes        
        // read value

        read_data(value, N, d, value_file);
        
    }
    int row_size=pattern_first.get_rows();
    double** part_query_first[row_size][d];
    double** part_query_last[row_size][d];
    {
        int* sendscounts= new int[num_procs];
        int* displs_first = new int[num_procs];
        int* displs_last = new int[num_procs];
        int sendcount=row_size*d;
        int first_off=pattern_first.start_row_id*d;
        int last_off=pattern_last.start_row_id*d;
        MPI_Gather(&sendcount, 1, MPI_INT, sendscounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&first_off, 1, MPI_INT, displs_first, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&last_off, 1, MPI_INT, displs_last, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(query, sendscounts, displs_first, MPI_DOUBLE, part_query_first, sendscounts[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(query, sendscounts, displs_last, MPI_DOUBLE, part_query_last, sendscounts[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    //For key communication
    double** part_key_first[pattern_first.col_ids.size()][d];
    double** part_key_last[pattern_last.col_ids.size()][d];
    double** part_val_first[pattern_first.col_ids.size()][d];
    double** part_val_last[pattern_last.col_ids.size()][d];
    {
        int* sendscounts= new int[num_procs];
        int* col_sizes= new int[num_procs];
        int col_size=pattern_first.col_ids.size();
        MPI_Gather(&col_size,1,MPI_INT,col_sizes,1,MPI_INT,0,MPI_COMM_WORLD);
        int** col_to_send;
        int total_cols=0;
        if (my_rank==0)
        {
            col_to_send=new int*[num_procs];
            for (size_t i = 0; i < num_procs; i++)
            {
                sendscounts[i]=col_sizes[i]*d;
                total_cols+=col_sizes[i];
                col_to_send[i]=new int[col_sizes[i]];
            }            
        }        
        MPI_Gatherv(pattern_first.col_ids.data(), col_size, MPI_INT, col_to_send, col_sizes, MPI_INT, 0, MPI_COMM_WORLD);
        for (size_t i = 0; i < num_procs; i++)
        {
            // MPI_Status status;
            // MPI_Request request_out1, request_in1;
            // MPI_Request request_out2, request_in2;
            if (my_rank==0)
            {
               double tmp_key[col_sizes[i]][d];
               double tmp_val[col_sizes[i]][d];
               for (size_t j = 0; j < col_sizes[i]; j++)
               {
                    for (size_t k = 0; k < d; k++)
                    {
                        tmp_key[j][k]=key[col_to_send[i][j]][k];
                        tmp_val[j][k]=value[col_to_send[i][j]][k];
                    }
                    
               }
               MPI_Isend(tmp_key, col_sizes[i]*d, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_REQUEST_NULL);
               MPI_Isend(tmp_val, col_sizes[i]*d, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_REQUEST_NULL);
            }
            MPI_Recv(part_key_first, col_sizes[i]*d, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Recv(part_val_first, col_sizes[i]*d, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);        
        }
    }
    

    for (size_t i = 0; i < N; i++)
    {
        delete[] query[i];
        delete[] key[i];
        delete[] value[i];
    }    
    delete[] value;
    delete[] key;
    delete[] query;
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