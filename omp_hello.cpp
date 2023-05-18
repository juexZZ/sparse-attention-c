#include<iostream>
#include<omp.h>

int main(){
    #pragma omp parallel
    {
    int capacity=omp_get_num_threads();
        #pragma omp for
        for(int i=0; i<capacity; i++){
            std::cout << "Hello World from thread " << omp_get_thread_num() << std::endl;
        }   
    }    
    return 0;
}