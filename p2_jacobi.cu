#include "residual.h"
#include <stdlib.h>
#define N 1024

__global__ 
void iterate(double* u_kp1, double* u_k, double* f, double *h_c){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double h = *h_c;
    if(row >=1 && row <= N && col >=1 && col <= N){
        u_kp1[row*(N+2) + col] = 1.0/4*(h*h*f[(row-1)*N + col-1] + u_k[(row-1)*(N+2) + col] + u_k[row*(N+2) + col - 1] + u_k[(row+1)*(N+2) + col] + u_k[row*(N+2) + col+1]);
        //printf("%f %f\n", u_kp1[row*(N+2) + col], 1.0/4*(h*h*f[(row-1)*N + col-1] + u_k[(row-1)*(N+2) + col] + u_k[row*(N+2) + col - 1] + u_k[(row+1)*(N+2) + col] + u_k[row*(N+2) + col+1]));
    }
}

__global__ 
void update(double* u_kp1, double* u_k){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >=1 && row <= N && col >=1 && col <= N){
        u_k[row*(N+2) + col] = u_kp1[row*(N+2) + col];
    }
}

void Check_CUDA_Error(const char *message){
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess) {
      fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
      exit(-1);
    }
}

int main(){
    double* u_k = (double*) malloc((N+2)*(N+2)*sizeof(double)); //u_k is used to store
    double* u_kp1 = (double*) malloc((N+2)*(N+2)*sizeof(double));
    double* f = (double*) malloc(N*N*sizeof(double));
    double h = 1.0/(N+1);

    double* u_k_d; 
    double* u_kp1_d;
    double* f_d;
    double* h_d;

    for(int i = 0; i < N+2; i++){
        for(int j = 0; j < N+2; j++){
            u_k[i*(N+2) + j] = rand() % 5;
        }
    }

    for (int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            f[i*N + j] = 1;
        }
    }

    cudaMalloc(&u_k_d, (N+2)*(N+2)*sizeof(double));
    cudaMalloc(&u_kp1_d, (N+2)*(N+2)*sizeof(double));
    cudaMalloc(&f_d, N*N*sizeof(double));
    cudaMalloc(&h_d, sizeof(double));

    cudaMemcpy(u_k_d, u_k, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
    Check_CUDA_Error("copy u_k failed");
    cudaMemcpy(f_d, f, N*N*sizeof(double), cudaMemcpyHostToDevice);
    Check_CUDA_Error("copy f failed");
    cudaMemcpy(h_d, &h, sizeof(double), cudaMemcpyHostToDevice);

    double initial_residual = residual(u_k, f, h, N);
    double current_residual = residual(u_k, f, h, N);
    int iteration = 0;

    dim3 block(32, 32);
    dim3 grid(32, 32);

    while(initial_residual/current_residual <=1e6 && iteration < 10000){
        iterate<<<grid,block>>>(u_kp1_d, u_k_d, f_d, h_d);
		cudaDeviceSynchronize();
		update<<<grid,block>>>(u_kp1_d, u_k_d);
		cudaDeviceSynchronize();
        iteration++;
        if(iteration %100 == 0){
            cudaMemcpy(u_k, u_k_d, (N+2)*(N+2)*sizeof(double), cudaMemcpyDeviceToHost); 
            current_residual = residual(u_k,f,h,N);
        }
    }
    cudaMemcpy(u_k, u_k_d, (N+2)*(N+2)*sizeof(double), cudaMemcpyDeviceToHost); 
    current_residual = residual(u_k,f,h,N);
    printf("inital_residual = %f last residual = %f\n", initial_residual ,current_residual);
}