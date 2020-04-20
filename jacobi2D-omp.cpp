#include "residual.h"

int main(int argc, char** argv) {
    for(int N = 100; N <= 1000; N = N + 100)
    {
    double* new_u = (double*) malloc((N+2)*(N+2)*sizeof(double)); //new_u is used to store
    double* u = (double*) malloc((N+2)*(N+2)*sizeof(double));
    double* f = (double*) malloc(N*N*sizeof(double));
    double* temp;  //used for swapping u and new_u;
    double h = 1.0/(N+1);

    for(int i = 0; i < N+2; i++){
        for(int j = 0; j < N+2; j++){
            u[i*(N+2) + j] = 0;
        }
    }

    for (int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            f[i*N + j] = 1;
        }
    }

    double initial_residual = residual(u,f,h,N);
    double current_residual = residual(u,f,h,N);
    int iteration = 0;
    Timer t;
    t.tic();
    while(initial_residual/current_residual <=1e6 && iteration < 1000){
        #if defined(_OPENMP)
        #pragma omp parallel for num_threads(thread_num)
        #endif
        for(int i = 1; i <= N; i++){
            for(int j = 1; j<= N; j++){
                new_u[i*(N+2)+j] = 1.0/4*(h*h*f[(i-1)*N + j-1] + u[(i-1)*(N+2) + j] + u[i*(N+2) + j - 1] + u[(i+1)*(N+2) + j] + u[i*(N+2) + j+1]);
            }
        }
        iteration++;
        temp = new_u;
        new_u = u;
        u = temp;
        current_residual = residual(u,f,h,N);
    }
    double time = t.toc();
    printf("Size: %d, iteration: %d, number of thread: %d, time: %f, initial %f,residual: %f\n", N, iteration, thread_num, time, initial_residual,current_residual);  
    free(new_u);
    free(u);
    free(f);
    }
    return 0;
}
