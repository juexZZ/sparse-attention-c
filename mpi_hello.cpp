// mpicxx -fopenmp -O3 mpi_hello.cpp -o mpi_hello
#include<iostream>
#include<omp.h>
#include<string>
#include<mpi.h>
using namespace std;
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

int main(int argc, char* argv[]){
    int num_procs, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    std::cout << "Hello World from process " << my_rank << std::endl;
    std::cout << "Number of processes: " << num_procs << std::endl;
    #pragma omp parallel
    {
    int capacity=omp_get_num_threads();
        #pragma omp for
        for(int i=0; i<capacity; i++){
            std::cout << "Hello World from thread " << omp_get_thread_num() << std::endl;
        }   
    } 
    int N=8192;
    int d=512;
    double *a=new double[N*d];
    double sum=0.0;
    string filename="data/n8192d512/key.txt";
    read_data(a, N, d, filename);
    std::cout<<"read data success"<<std::endl;
    double tt1=omp_get_wtime();
    #pragma omp parallel for reduction(+:sum)
    for (long long i=0; i<N*d; i++){
        sum+=a[i];
    }
    double tt2=omp_get_wtime();
    std::cout<<"omp time: "<<tt2-tt1<<std::endl;
    MPI_Finalize();
    return 0;
}