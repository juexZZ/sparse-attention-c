// sparse attention in cpp
// arthur: Juexiao Zhang, Yiwei Shao
// May 2023
#include <math.h>
#include <vector>
#include <set>
#include <string>
using namespace std;

class SparsePattern
{
    public:
        int N,d;
        int start_row_id;
        int end_row_id;    
        vector<int> col_ids;
        vector<vector<int>>col_ids_row;
        vector<int> get_row_id(){
            vector<int> row_ids;
            for(int i=start_row_id; i<end_row_id; i++){
                row_ids.push_back(i);
            }
            return row_ids;
        };
        SparsePattern(int N_, int d_ ,int start, int end, vector<int> col_ids_){
            N = N_;
            d = d_;
            start_row_id = start;
            end_row_id = end;
            col_ids = col_ids_;};
        int get_rows(){
            return start_row_id-end_row_id;
        }
        SparsePattern(){};
        ~SparsePattern(){};
};



template <typename T>
T* switch_order(T* input, int num, int dim){
    // switch the order of the matrix, from row major to column major, or vice versa
    // input: the input matrix, num x dim
    // output: the output matrix, dim x num
    // num: number of rows
    // dim: dimension of each row
    T* output = new T[num*dim];
    for(int i=0; i<num; i++){
        for(int j=0; j<dim; j++){
            output[j*num+i] = input[i*dim+j];
        }
    }
    return output;
}

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

void get_fixed_sparse_idx(int row_id, vector<int>& col_ids, set<int>& set_total_col_ids, int l, int c = 0){
    // get the sparse indexes of a row
    // row_id: the row id (starting from 0)
    // col_ids: the column ids of the non zero elements
    int m = (int)(row_id+1)/l;
    for(int idx = 1; idx <= row_id+1; idx++){
        if((int)idx/l == m){ // pattern A1 in the paper
            col_ids.push_back(idx-1);
            set_total_col_ids.insert(idx-1);
        }
        else if( idx % l == 0 || (idx%l) >= l-c){ // pattern A2 in the paper
            col_ids.push_back(idx-1);
            set_total_col_ids.insert(idx-1);
        }
    }
}

void get_strided_sparse_idx(int row_id, vector<int>& col_ids, set<int>& set_total_col_ids, int l){
    // get the sparse indexes of a row
    // row_id: the row id (starting from 0)
    // col_ids: the column ids of the non zero elements
    // c: the context window
    for(int idx = 1; idx <= row_id+1; idx++){
        if((row_id - idx)%l == 0 ){ // pattern A2 in the paper
            col_ids.push_back(idx-1);
            set_total_col_ids.insert(idx-1);
        }
        else if (idx >= row_id - l){ // pattern A1
            col_ids.push_back(idx-1);
            set_total_col_ids.insert(idx-1);
        }
        
    }
}

void get_row_share(int rank, int num_procs, int N, SparsePattern& pattern_first, SparsePattern& pattern_last){
    // get each process's share of rows
    // rank: the rank of the process
    // row_ids: the row ids of the rows that this process holds
    int start_id=N/num_procs/2*rank;
    int end_id=N/num_procs/2*(rank+1);
    pattern_first.start_row_id = start_id;
    pattern_first.end_row_id = end_id;
    pattern_last.start_row_id = N-end_id;
    pattern_last.end_row_id = N-start_id;
}

void get_column_share(int rank, int N, int d,SparsePattern& tmp_pat, int pattern, int context_l, int fixed_c)
    {
    // get each process's share of columns
    // rank: the rank of the process
    set<int> set_total_col_ids;
    if(pattern==0)
    {
        //Strided Pattern
        tmp_pat.col_ids_rows.resize(tmp_pat.end_row_id-tmp_pat.start_row_id);
        for(int ri=tmp_pat.start_row_id; ri<tmp_pat.end_row_id; ri++){
            // index the col_ids_rows accordingly
            get_strided_sparse_idx(ri, tmp_pat.col_ids_rows[ri-tmp_pat.start_row_id], set_total_col_ids, context_l);
        }
    }
    else if(pattern==1)
    {   
        //Fixed Pattern
        tmp_pat.col_ids_rows.resize(tmp_pat.end_row_id-tmp_pat.start_row_id);        
        for(int ri=tmp_pat.start_row_id; ri<tmp_pat.end_row_id; ri++)
        {
            // index the col_ids_rows accordingly
            get_fixed_sparse_idx(ri, tmp_pat.col_ids_rows[ri-tmp_pat.start_row_id], set_total_col_ids, context_l);
        }
    }
    else
    {
        std::throw runtime_error("Invalid Pattern");
    }
    // TODO: sort it before returning?
    tmp_pat.col_ids.assign(set_total_col_ids.begin(), set_total_col_ids.end());
    tmp_pat.col_ids.sort();
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
