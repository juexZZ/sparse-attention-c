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
    int threads=4;
    int N = 32;
    int d = 8;
    int context_l = 64; // paper l
    int fixed_c = 0; // paper c
    int num_procs, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    string data_dir=argv[1];
    // threads=atoi(argv[2]);
    string query_file = "data/"+data_dir+"/query.txt";
    string key_file = "data/"+data_dir+"/key.txt";
    string value_file = "data/"+data_dir+"/value.txt";
    std::size_t n_found=data_dir.find("n");
    std::size_t d_found=data_dir.find("d");
    N=stoi(data_dir.substr(n_found+1,d_found-n_found-1));
    d=stoi(data_dir.substr(d_found+1));
    bool is_debug=false;
    if(is_debug){
        std::cout<<"N: "<<N<<endl;
        std::cout<<"d: "<<d<<endl;
        std::cout<<"threads: "<<threads<<endl;
    }
    if (my_rank==0)
    {
        std::cout<<data_dir<<endl;
    }
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Rank %d/%d running on %s.\n", my_rank, num_procs, processor_name);

// #ifdef _OPENMP
//     omp_set_num_threads(threads);
// #endif
    // read data and distribute to other processes
    // each process hold: part_query_first, part_key_first, part_value_first, part_query_last, part_key_last, part_value_last
    // to keep thread load balanced

    // SparsePattern pattern_first(N,d);
    // SparsePattern pattern_last(N,d);
    SparsePattern pattern_proc(N, d);

    int pattern=1; //Two patterns, 0 is strided, 1 is fixed.
    get_row_share(my_rank,num_procs, N, pattern_proc);
    if (is_debug)
    {
       std::cout<<"get row share done"<<std::endl;  
    }
    get_column_share(my_rank, N, d, pattern_proc, pattern, context_l, fixed_c);
    if (is_debug)
    {
       std::cout<<"get column share done"<<std::endl;  
    }
    // get_column_share(my_rank, N, d, pattern_last, pattern, context_l, fixed_c);
    pattern_proc.build_inverse_col_id();
    if (is_debug)
    {
       std::cout<<"build inverse col id done"<<std::endl;  
    }
    // pattern_last.build_inverse_col_id();

    // For query communication
    double* query= new double[N*d];
    double* key=new double[N*d]; // Here key is transpose
    double* value=new double[N*d];
    if (is_debug)
    {
       std::cout<<"Ready to read"<<std::endl;  
    }
    if (my_rank==0){    
       
        read_data(query, N, d, query_file);
       
        read_data(key, N, d, key_file);

        read_data(value, N, d, value_file);
        
    }
    if (is_debug)
    {
       std::cout<<"read data done"<<std::endl;  
    }
    
    double init_time=MPI_Wtime();
    int row_size=pattern_proc.get_rows();
    // double part_query_first[row_size*d];
    // double part_query_last[row_size*d];
    double* part_query=new double[row_size*d];
    int row_size_front = pattern_proc.get_rows_front();
    int row_size_back = pattern_proc.get_rows_back();
    double* part_query_back = part_query+row_size_front*d;
    int* row_sizes_front= new int[num_procs];
    int* row_sizes_back = new int[num_procs];
    int* displs_front = new int[num_procs];
    int* displs_back = new int[num_procs];
    int sendcount_front=row_size_front*d;
    int sendcount_back = row_size_back*d;
    {
        int* sendcounts_front= new int[num_procs];
        int* sendcounts_back = new int[num_procs];
        int front_off=pattern_proc.start_row_id_front*d;
        int back_off=pattern_proc.start_row_id_back*d;
        MPI_Gather(&row_size_front, 1, MPI_INT, row_sizes_front, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&row_size_back, 1, MPI_INT, row_sizes_back, 1, MPI_INT, 0, MPI_COMM_WORLD);        
        MPI_Gather(&front_off, 1, MPI_INT, displs_front, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&back_off, 1, MPI_INT, displs_back, 1, MPI_INT, 0, MPI_COMM_WORLD);
        for (size_t i = 0; i < num_procs; i++)
        {
            sendcounts_front[i]=row_sizes_front[i]*d;
            sendcounts_back[i] = row_sizes_back[i]*d;
        }
        MPI_Scatterv(query, sendcounts_front, displs_front, MPI_DOUBLE, part_query, sendcount_front, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(query, sendcounts_back, displs_back, MPI_DOUBLE, part_query_back, sendcount_back, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        delete[] sendcounts_front;
        delete[] sendcounts_back;
    }
    if (is_debug)
    {
       std::cout<<"query communication done"<<std::endl;  
    }
    //For key communication
    double* part_key = new double[pattern_proc.col_ids.size()*d];
    double* part_val = new double[pattern_proc.col_ids.size()*d];
    // double* part_key_first = new double[pattern_first.col_ids.size()*d];
    // double* part_key_last = new double[pattern_last.col_ids.size()*d];
    // double* part_val_first = new double[pattern_first.col_ids.size()*d];
    // double* part_val_last = new double[pattern_last.col_ids.size()*d];
    {
        int* sendcounts= new int[num_procs];
        int* col_sizes= new int[num_procs];
        int col_size=pattern_proc.col_ids.size();
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
              
        MPI_Gatherv(pattern_proc.col_ids.data(), col_size, MPI_INT, mpi_col_to_send, sendcounts, offsets.data(), MPI_INT, 0, MPI_COMM_WORLD);        
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
        }
        MPI_Status status;
        MPI_Request request_out1[num_procs];
        MPI_Request request_out2[num_procs];
        if (is_debug)
        {
            std::cout<<"col info communication done"<<std::endl;
        }
        if (my_rank==0)
        {
            for (size_t i = 0; i < num_procs; i++)
            {
                double **tmp_key;
                double **tmp_val;
                tmp_key=create2DArray<double>(col_sizes[i],d);
                tmp_val=create2DArray<double>(col_sizes[i],d);
                for (size_t j = 0; j < col_sizes[i]; j++)
                {
                    for (size_t k = 0; k < d; k++)
                    {
                        tmp_key[j][k]=key[col_to_send[i][j]*d+k];
                        tmp_val[j][k]=value[col_to_send[i][j]*d+k];
                    }
                    
                }
                double *send_key=reinterpret_cast<double *>(&tmp_key[0][0]);
                double *send_val=reinterpret_cast<double *>(&tmp_val[0][0]);
                MPI_Isend(send_key, col_sizes[i]*d, MPI_DOUBLE, i, 2*i, MPI_COMM_WORLD, &request_out1[i]);
                MPI_Isend(send_val, col_sizes[i]*d, MPI_DOUBLE, i, 2*i+1, MPI_COMM_WORLD, &request_out2[i]);
            }
        }
        MPI_Recv(part_key, col_size*d, MPI_DOUBLE, 0, 2*my_rank, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(part_val, col_size*d, MPI_DOUBLE, 0, 2*my_rank+1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if (my_rank==0)
        {
            for (size_t i = 0; i < num_procs; i++)
            {
                MPI_Request_free(&request_out1[i]);
                MPI_Request_free(&request_out2[i]);
            }
        }        

    }
    if (is_debug)
    {
       std::cout<<"key value communication done"<<std::endl;  
    }
    double comm_time=MPI_Wtime();
    std::cout<<"Communication time "<<comm_time-init_time<<std::endl;
    
    // sparse attention
    double** attn_w = new double*[pattern_proc.get_rows()];
    // double** attn_w_first = new double*[pattern_first.get_rows()];
    // double** attn_w_last = new double*[pattern_last.get_rows()];

    #pragma omp parallel for 
    for(int r=0; r<pattern_proc.get_rows(); r++){
        attn_w[r] = new double[pattern_proc.col_ids_row[r].size()];
        row_sparse_attention(part_query+r*d, part_key, attn_w[r], pattern_proc, r);
    }

    // for (int r = 0; r < row_ids_last.size(); r++)
    // {
    //     attn_w_last[r] = new double[pattern_last.col_ids_row[r].size()];
    //     row_sparse_attention(part_query_last[r], part_key_last, attn_w_last[r], pattern_last.col_ids_row[r].size(), d);
    // }

    // attention times value
    // initialize result matrix as the same size as value, initialize with 0
    vector<double> part_result(pattern_proc.get_rows()*d, 0);
    attn_weight_value(attn_w, part_val, part_result, pattern_proc);
    // attn_weight_value(attn_w_last, part_val_last, result_last, pattern_last);
    //communicate results, just gather
    if (is_debug)
    {
       std::cout<<"attention done"<<std::endl;  
    }
    double *result_1d = new double[N*d];
    {
        int* sendcounts_front= new int[num_procs];
        int* sendcounts_back = new int[num_procs];
        for (size_t i = 0; i < num_procs; i++)
        {
            sendcounts_front[i]=row_sizes_front[i]*d;
            sendcounts_back[i] =row_sizes_back[i]*d;
        }
        // idspls only significant to the root
        MPI_Gatherv(part_result.data(), sendcount_front, MPI_DOUBLE, result_1d, sendcounts_front, displs_front, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(part_result.data()+sendcount_front, sendcount_back, MPI_DOUBLE, result_1d, sendcounts_back, displs_back, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        delete[] sendcounts_front;
        delete[] sendcounts_back;
    }
    double cal_time=MPI_Wtime();
    std::cout<<"Calculation time "<<cal_time-comm_time<<std::endl;
    
    // save_data(result_1d, N, d, "data/"+data_dir+"/result.txt");
    // free attn_w
    for(int r=0; r<pattern_proc.get_rows(); r++){
        delete[] attn_w[r];
    }
    
    delete[] attn_w;
    // free the rest
    delete[] part_key;
    delete[] part_val;
    delete[] row_sizes_front;
    delete[] row_sizes_back;
    delete[] displs_front;
    delete[] displs_back;
    
    MPI_Finalize();

}