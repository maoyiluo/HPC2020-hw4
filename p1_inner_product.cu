#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#define blockSize 1024

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
static __inline__ __device__ double atomicAdd(double* address, double val)
{
        unsigned long long int* address_as_ull =
                (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;

        do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed,
                                __double_as_longlong(val +
                                        __longlong_as_double(assumed)));

                // Note: uses integer comparison to avoid hang in case of NaN 
                // (since NaN != NaN)
        } while (assumed != old);

        return __longlong_as_double(old);
}
#endif

double vec_inner_product(double* a, double* b, long N){
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
    __shared__ double temp[blockSize];
    if (idx < N) temp[threadIdx.x] = a[idx] * b[idx];
    int i = blockSize/2;
    __syncthreads();
    while(i != 0){
        if(threadIdx.x < i){
            temp[threadIdx.x] += temp[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if(threadIdx.x == 0){
        atomicAdd(res, temp[0]);
    }
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
    res_ref = vec_inner_product(x, y, N);
    printf("CPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

    double *x_d, *y_d, *res_d;
    cudaMalloc(&x_d, N*sizeof(double));
    Check_CUDA_Error("malloc x failed");
    cudaMalloc(&y_d, N*sizeof(double));
    cudaMalloc(&res_d, sizeof(double));
    Check_CUDA_Error("memset failed");

    tt = omp_get_wtime();
    cudaMemcpy(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(res_d, 0, sizeof(double), cudaMemcpyHostToDevice);
    vec_inner_product_kernel<<<N/blockSize, blockSize>>>(res_d, x_d, y_d, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&res, res_d, sizeof(double), cudaMemcpyDeviceToHost);
    printf("GPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

    double err = res_ref-res;
    printf("Error = %f\n",err);


    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(res_d);

    free(x);
    free(y);

    return 0;
  }
