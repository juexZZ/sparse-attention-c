// sparse attention in cpp
// arthur: Juexiao Zhang, Yiwei Shao
// May 2023
#include <math.h>
#include <vector>
#include <string>
using namespace std;

void inplace_softmax(double* row_vec, int length){
    // softmax by row
    // row vec: a row of the attention matrix, only the non zero elements
    double sum = 0;
    for(int i=0; i<length; i++){
        double v = exp(row_vec[i]);
        sum += v;
        row_vec[i] = v;
    }
    // normalize
    for(int i=0; i<length; i++){
        row_vec[i] = row_vec[i] / sum;
    }
}

// obtain u and v by indexing the matrices U and V via sparse index
double scaled_dot(double* query, double* key, int dim){
    // dot product of u and v, scaled
    double prod = 0;
    for (int i=0; i<dim; i++){
        prod += query[i] * key[i];
    }
    return prod / sqrt(dim);
}   

void row_sparse_attention(double* query, double** keys, double* res, int num, int dim){
    // get sparse attention matrix by each row
    // keys is num x d, n equals to the number of nonzeros
    // predefine num and obtain Vs according to sparse indexes
    for(int i=0; i<num; i++){
        res[i] = scaled_dot(query, keys[i], dim);
    }
    inplace_softmax(res, num);
}

void weight_value(double* attn_w, double* value, double* res){
    // apply attention weights to value columns, results in res
    // attn_w: the attention matrix, sparse
    // value: column of value V
    // this is sparse matrix times column vector

}

void get_row_sparse_idx(int row_id, vector<int>& col_ids){
    // get the sparse indexes of a row
    // row_id: the row id
    // col_ids: the column ids of the non zero elements

}

void get_row_share(int rank, int N, vector<int>& row_ids){
    // get each process's share of rows
    // rank: the rank of the process
    // row_ids: the row ids of the rows that this process holds
    // NOTE: currently each process take two rows, from both side to the middle
    row_ids.push_back(rank);
    if (rank != N - rank - 1)
        row_ids.push_back(N - rank - 1);
}

void get_column_share(int rank, int N, int d, vector<int>& row_ids, 
                    vector<int>& total_col_ids, 
                    vector<vector<int>>& col_ids_row){
    // get each process's share of columns
    // rank: the rank of the process

}

// ********************* I/O *****************************
// process 0 read data Q, K, V from separate files
// query, key and value have same size: num x dim
int read_data(double* data, int num, int dim, string filename){
    // read query data from file filename
    // data in the file is in txt format, spaced by \t and \n, each row has dim elements, num rows in total.
    // query: the query matrix, num x dim
    // num: number of rows
    // dim: dimension of each row
    // store in a 2d array
    // return 0 if success, -1 if fail
    FILE* fp = fopen(filename.c_str(), "r");
    if(fp == NULL){
        printf("Error: cannot open file %s\n", filename.c_str());
        return -1;
    }
    for(int i=0; i<num; i++){
        for(int j=0; j<dim; j++){
            fscanf(fp, "%lf", &data[i*dim+j]);
        }
    }
    fclose(fp);
    return 0;
}
