#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

double vec_inner_product(const double* a, const double* b, long N){
  double sum = 0;
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i = 0; i < N; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

__global__
void vec_inner_product_kernel(double* res, const double* a, const double* b, long N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  double temp = 0;
  if (idx < N) temp = a[idx] + b[idx];
  atomicAdd(sum,&temp);
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main() {
    long N = (1UL<<25); // 2^25
  
    double* x = (double*) malloc(N * sizeof(double));
    double* y = (double*) malloc(N * sizeof(double));
    double res = 0;
    double res_ref = 0;

    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) {
      x[i] = i+2;
      y[i] = 1.0/(i+1);
    }
  
    double tt = omp_get_wtime();
    res_ref = vec_inner_product(res_ref, x, y, N);
    printf("CPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  
    double *x_d, *y_d, *res_d;
    cudaMalloc(&x_d, N*sizeof(double));
    Check_CUDA_Error("malloc x failed");
    cudaMalloc(&y_d, N*sizeof(double));
    cudaMalloc(&res_d, sizeof(double));
  
    tt = omp_get_wtime();
    cudaMemcpy(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
    vec_add_kernel<<<N/1024,1024>>>(res_d, x_d, y_d, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&res, res_d, N*sizeof(double), cudaMemcpyDeviceToHost);
    printf("GPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  
    double err = res_ref-res;
    printf("Error = %f\n", err);
  
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(res_d);
  
    free(x);
    free(y);
  
    return 0;
  }
  