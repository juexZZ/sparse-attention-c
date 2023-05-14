// sparse attention in cpp
// arthur: Juexiao Zhang, Yiwei Shao
// May 2023
#include <math.h>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
using namespace std;
#include <exception>

template <typename T>
T** create2DArray(unsigned nrows, unsigned ncols, const T& val = T())
{
   if (nrows == 0)
        throw std::invalid_argument("number of rows is 0");
   if (ncols == 0)
        throw std::invalid_argument("number of columns is 0");
   T** ptr = nullptr;
   T* pool = nullptr;
   try
   {
       ptr = new T*[nrows];  // allocate pointers (can throw here)
       pool = new T[nrows*ncols]{val};  // allocate pool (can throw here)

       // now point the row pointers to the appropriate positions in
       // the memory pool
       for (unsigned i = 0; i < nrows; ++i, pool += ncols )
           ptr[i] = pool;

       // Done.
       return ptr;
   }
   catch (std::bad_alloc& ex)
   {
       delete [] ptr; // either this is nullptr or it was allocated
       throw ex;  // memory allocation error
   }
}

template <typename T>
void delete2DArray(T** arr)
{
   delete [] arr[0];  // remove the pool
   delete [] arr;     // remove the pointers
}
class SparsePattern
{
    public:
        int N,d;
        int start_row_id_front;
        int end_row_id_front;
        int start_row_id_back;
        int end_row_id_back;    
        vector<int> col_ids;
        vector<int> row_ids;
        vector<int> inverse_col_ids;
        vector<vector<int>>col_ids_row;
        void build_inverse_col_id(){
            for(int i=0; i<col_ids.size(); i++){
                inverse_col_ids[col_ids[i]] = i;
            }
        };
        void build_row_ids(){
            for(int ri=start_row_id_front; ri<end_row_id_front; ri++){
                row_ids.push_back(ri);
            }
            for(int ri=start_row_id_back; ri<end_row_id_back; ri++){
                row_ids.push_back(ri);
            }
        };
        SparsePattern(int N_, int d_ ): inverse_col_ids(N_,-1){
            N = N_;
            d = d_;
            // inverse_col_ids.resize(N_);
            // std::fill(inverse_col_ids.begin(), inverse_col_ids.end(), -1);
            };
        int get_rows_front(){
            return end_row_id_front-start_row_id_front;
        }
        int get_rows_back(){
            return end_row_id_back-start_row_id_back;
        }
        int get_rows(){
            return row_ids.size(); // should equal to get_rows_front() + get_rows_back()
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
        prod += *(query+i) * *(key+i);
    }
    return prod / sqrt(dim);
}   

void row_sparse_attention(double* query, double* keys, double* res, SparsePattern& pattern, int r){
    // get sparse attention matrix by each row
    // keys is total_col_ids x d,
    // predefine num and obtain Vs according to sparse indexes
    int num = pattern.col_ids_row[r].size();
    int dim = pattern.d;
    for(int i=0; i<num; i++){
        int global_col_idx = pattern.col_ids_row[r][i];
        int local_col_idx = pattern.inverse_col_ids[global_col_idx];
        res[i] = scaled_dot(query, keys+local_col_idx*dim, dim);
    }
    inplace_softmax(res, num);
}

// obtain results by weight the value matrix with the attention matrix
// once again follow the order of the attention matrix rows
void row_attn_weight_value(double* attn_w_row, double* value, 
    int res_offset, vector<double>& res,
    vector<int>& col_ids_row, vector<int>& inverse_col_ids, int d){
    // apply attention weights to value columns, results in res
    // attn_w: one row of the attention matrix, sparse, only have the nonzero values
    // value: chunk of value handled by this process, num x dim
    // res: only populate the result's row corresponding to this attn_w_row: 1 x dim
    // NOTE: index using double pointer method or index inverse hash map
    for(int i=0; i<col_ids_row.size(); i++){
        int vid = inverse_col_ids[col_ids_row[i]]; // get the index of the value in the part_value
        for(int j=0; j<d; j++){
            res[j+res_offset] += attn_w_row[i] * value[vid*d+j]; // transpose the value matrix? or  change loop order?
        }
    }
}

void attn_weight_value(double** attn_w, double* value, vector<double>& res, SparsePattern& pattern){
    int row_size = pattern.get_rows();
    #pragma omp parallel for
    for(int ri=0; ri<row_size; ri++){
        // int row_id = pattern.start_row_id + ri;
        row_attn_weight_value(attn_w[ri], value, ri*pattern.d, res, pattern.col_ids_row[ri], pattern.inverse_col_ids, pattern.d);
    }
}

void get_fixed_sparse_idx(int row_id, vector<int>& col_ids, set<int>& set_total_col_ids, int l, int c = 0){
    // get the sparse indexes of a row
    // row_id: the row id (starting from 0)
    // col_ids: the column ids of the non zero elements
    int m = (int)(row_id+1)/l;
    for(int idx = 1; idx < row_id+1; idx++){
        if((int)idx/l == m){ // pattern A1 in the paper
            col_ids.push_back(idx-1);
            set_total_col_ids.insert(idx-1);
        }
        else if( idx % l == 0 || (idx%l) >= l-c){ // pattern A2 in the paper
            col_ids.push_back(idx-1);
            set_total_col_ids.insert(idx-1);
        }
    }
    col_ids.push_back(row_id); // must have the diagonal element
    set_total_col_ids.insert(row_id);
}

void get_strided_sparse_idx(int row_id, vector<int>& col_ids, set<int>& set_total_col_ids, int l){
    // get the sparse indexes of a row
    // row_id: the row id (starting from 0)
    // col_ids: the column ids of the non zero elements
    // c: the context window
    for(int idx = 1; idx < row_id+1; idx++){
        if((row_id - idx)%l == 0 ){ // pattern A2 in the paper
            col_ids.push_back(idx-1);
            set_total_col_ids.insert(idx-1);
        }
        else if (idx >= row_id - l){ // pattern A1
            col_ids.push_back(idx-1);
            set_total_col_ids.insert(idx-1);
        }
    }
    col_ids.push_back(row_id); // must have the diagonal element
    set_total_col_ids.insert(row_id);
}

void get_row_share(int rank, int num_procs, int N, SparsePattern& pattern_proc){
    // get each process's share of rows
    // rank: the rank of the process
    // row_ids: the row ids of the rows that this process holds
    int start_id=N/num_procs/2*rank;
    int end_id=N/num_procs/2*(rank+1);
    pattern_proc.start_row_id_front = start_id;
    pattern_proc.end_row_id_front = end_id;
    pattern_proc.start_row_id_back = N - end_id;
    pattern_proc.end_row_id_back = N - start_id;
    pattern_proc.build_row_ids();
}

void get_column_share(int rank, int N, int d,SparsePattern& tmp_pat, int pattern, int context_l, int fixed_c)
    {
    // get each process's share of columns
    // rank: the rank of the process
    set<int> set_total_col_ids;
    if(pattern==0)
    {
        //Strided Pattern
        tmp_pat.col_ids_row.resize(tmp_pat.get_rows());
        for(int i = 0; i < tmp_pat.get_rows(); i++)
        {
            get_strided_sparse_idx(tmp_pat.row_ids[i], tmp_pat.col_ids_row[i], set_total_col_ids, context_l);   
        }
    }
    else if(pattern==1)
    {   
        //Fixed Pattern
        tmp_pat.col_ids_row.resize(tmp_pat.get_rows());        
        for(int i=0; i < tmp_pat.get_rows(); i++)
        {
            // index the col_ids_rows accordingly
            get_fixed_sparse_idx(tmp_pat.row_ids[i], tmp_pat.col_ids_row[i], set_total_col_ids, context_l, fixed_c);
        }
    }
    // sort it before returning
    tmp_pat.col_ids.assign(set_total_col_ids.begin(), set_total_col_ids.end());
    sort(tmp_pat.col_ids.begin(), tmp_pat.col_ids.end());
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
            fscanf(fp, "%lf", data+i*dim+j);
        }
    }
    fclose(fp);
    return 0;
}

int save_data(double* data,int num, int dim, string filename){
    // save data to file
    // data: the data to be saved
    // num: number of rows
    // dim: dimension of each row
    // filename: the file name to be saved
    // return 0 if success, -1 if fail
    FILE* fp = fopen(filename.c_str(), "w");
    if(fp == NULL){
        printf("Error: cannot open file %s\n", filename.c_str());
        return -1;
    }
    for(int i=0; i<num; i++){
        for(int j=0; j<dim; j++){
            fprintf(fp, "%lf\t", data[i*dim+j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    return 0;
}