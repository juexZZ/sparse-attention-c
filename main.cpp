// main cpp for sparse attention
// author: Juexiao Zhang, Yiwei Shao
// May 2023
#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "utils.h"
#include <unistd.h>


using namespace std;

int main(int argc, char* argv[]){
    // read query, key and value from seperate files
    int N = 32;
    int d = 8;
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

    SparsePattern pattern_first(N,d);
    SparsePattern pattern_last(N,d);

    int pattern=0; //Two patterns, 0 is strided, 1 is fixed.
    get_row_share(my_rank,num_procs, N, pattern_first, pattern_last);
    get_column_share(my_rank, N, d, pattern_first, pattern, context_l, fixed_c);
    get_column_share(my_rank, N, d, pattern_last, pattern, context_l, fixed_c);
    pattern_first.build_inverse_col_id();
    pattern_last.build_inverse_col_id();

    // For query communication
    double query[N*d];
    double key[N*d]; // Here key is transpose
    double value[N*d];
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
    double part_query_first[row_size*d];
    double part_query_last[row_size*d];
    int* row_sizes= new int[num_procs];
    int* displs_first = new int[num_procs];
    int* displs_last = new int[num_procs];
    {
        int* sendcounts= new int[num_procs];
        int sendcount=row_size*d;
        int first_off=pattern_first.start_row_id*d;
        int last_off=pattern_last.start_row_id*d;
        MPI_Gather(&row_size, 1, MPI_INT, row_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);        
        MPI_Gather(&first_off, 1, MPI_INT, displs_first, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&last_off, 1, MPI_INT, displs_last, 1, MPI_INT, 0, MPI_COMM_WORLD);
        for (size_t i = 0; i < num_procs; i++)
        {
            sendcounts[i]=row_sizes[i]*d;
        }
        MPI_Scatterv(query, sendcounts, displs_first, MPI_DOUBLE, part_query_first, sendcounts[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(query, sendcounts, displs_last, MPI_DOUBLE, part_query_last, sendcounts[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        delete[] sendcounts;
    }
    std::cout<<"Scatter query done"<<std::endl;
    //For key communication
    double* part_key_first = new double[pattern_first.col_ids.size()*d];
    double* part_key_last = new double[pattern_last.col_ids.size()*d];
    double* part_val_first = new double[pattern_first.col_ids.size()*d];
    double* part_val_last = new double[pattern_last.col_ids.size()*d];
    {
        int* sendcounts= new int[num_procs];
        int* col_sizes= new int[num_procs];
        int col_size=pattern_first.col_ids.size();
        MPI_Gather(&col_size,1,MPI_INT,col_sizes,1,MPI_INT,0,MPI_COMM_WORLD);
        int** col_to_send;
        int* mpi_col_to_send;
        int total_cols=0;
        if (my_rank==0)
        {
            col_to_send=new int*[num_procs];
            for (size_t i = 0; i < num_procs; i++)
            {
                sendcounts[i]=col_sizes[i]*d;
                total_cols+=col_sizes[i];
                col_to_send[i]=new int[col_sizes[i]];
            }
            mpi_col_to_send=new int[total_cols];            
        } 
        vector<int> offsets(num_procs,0);
        for (size_t i = 1; i < num_procs; i++)
        {
            offsets[i]=offsets[i-1]+col_sizes[i-1];
        }
              
        MPI_Gatherv(pattern_first.col_ids.data(), col_size, MPI_INT, mpi_col_to_send, sendcounts, offsets.data(), MPI_INT, 0, MPI_COMM_WORLD);
        std::cout<<"Gather key done"<<std::endl;
        if (my_rank==0)
        {
            int offset=0;
            for (size_t i = 0; i < num_procs; i++)
            {
                for (size_t j = 0; j < col_sizes[i]; j++)
                {
                    col_to_send[i][j]=mpi_col_to_send[offset+j];
                }
                offset+=col_sizes[i];
            }
            delete[] mpi_col_to_send;
            std::cout<<"offset"<<std::endl;
        }
        MPI_Status status;
        MPI_Request request_out1, request_in1;
        MPI_Request request_out2, request_in2;
        if (my_rank==0)
        {
            for (size_t i = 0; i < num_procs; i++)
            {
                double tmp_key[col_sizes[i]][d];
                double tmp_val[col_sizes[i]][d];
                for (size_t j = 0; j < col_sizes[i]; j++)
                {
                    for (size_t k = 0; k < d; k++)
                    {
                        tmp_key[j][k]=key[col_to_send[i][j]*d+k];
                        tmp_val[j][k]=value[col_to_send[i][j]*d+k];
                    }
                    
                }
                MPI_Isend(tmp_key, col_sizes[i]*d, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &request_out1);
                MPI_Isend(tmp_val, col_sizes[i]*d, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &request_out2);
                std::cout<<"send"<<std::endl;    
            }
        }
        MPI_Recv(part_key_first, col_size*d, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(part_val_first, col_size*d, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
        std::cout<<"recv"<<std::endl;       
    }
    
    // sparse attention
    double** attn_w_first = new double*[pattern_first.get_rows()];
    double** attn_w_last = new double*[pattern_last.get_rows()];

    
    for(int r=0; r<pattern_first.get_rows(); r++){
        attn_w_first[r] = new double[pattern_first.col_ids_row[r].size()];
        row_sparse_attention(part_query_first+r*d, part_key_first, attn_w_first[r], pattern_first.col_ids_row[r].size(), d);
    }

    // for (int r = 0; r < row_ids_last.size(); r++)
    // {
    //     attn_w_last[r] = new double[pattern_last.col_ids_row[r].size()];
    //     row_sparse_attention(part_query_last[r], part_key_last, attn_w_last[r], pattern_last.col_ids_row[r].size(), d);
    // }

    // attention times value
    // initialize result matrix as the same size as value, initialize with 0
    vector<double> result_first(pattern_first.get_rows()*d, 0);
    vector<double> result_last(pattern_last.get_rows()*d, 0);
    attn_weight_value(attn_w_first, part_val_first, result_first, pattern_first);
    // attn_weight_value(attn_w_last, part_val_last, result_last, pattern_last);
    //communicate results, just gather
    // TODO
    double result[N][d];
    {
        int* sendcounts= new int[num_procs];
        for (size_t i = 0; i < num_procs; i++)
        {
            sendcounts[i]=row_sizes[i]*d;
        }
        MPI_Gatherv(result_first.data(),result_first.size(), MPI_DOUBLE, result, sendcounts, displs_first, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // MPI_Gatherv(result_last.data(),result_last.size(), MPI_DOUBLE, result, sendcounts, displs_last, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        delete[] sendcounts;
    }
    // free attn_w
    for(int r=0; r<pattern_first.get_rows(); r++){
        delete[] attn_w_first[r];
    }
    delete[] attn_w_first;
    // free the rest
    delete[] part_key_first;
    delete[] part_val_first;
    delete[] part_key_last;
    delete[] part_val_last;
    delete[] row_sizes;
    delete[] displs_first;
    delete[] displs_last;

    MPI_Finalize();

}