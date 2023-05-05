// sparse attention in cpp
#include <math.h>

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

}

