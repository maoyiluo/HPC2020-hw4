#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#define blockSize 8

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

void matrix_vec_mul(double* res, double*a, double *b, long N){
   #pragma omp parallel for schedule(static)
    for(long i = 0; i < N; i++){
        double temp = 0;
        for(long j = 0; j < N; j++){
            temp += a[j+i*N]*b[j];
        }
      //#pragma omp atomic
	res[i] += temp;
    }
}

__global__
void matrix_vec_product_kernel(double* res, const double* a, const double* b, long N){
    long col = blockIdx.x * blockDim.x + threadIdx.x;
    long row = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ double temp[blockSize][blockSize];
    if (row < N && col < N) temp[threadIdx.y][threadIdx.x] = a[row*N +col] * b[col];
    __syncthreads();

	for(int y = 0; y < blockSize; y++){
        int i = blockSize/2;
        while(i != 0){
            if(threadIdx.x < i){
                temp[y][threadIdx.x] += temp[y][threadIdx.x + i];
            }
            __syncthreads();
            i /= 2;
        }
    }
    __syncthreads();
    if(threadIdx.x == 0 && threadIdx.y == 0){
        for(int y = 0; y < blockSize; y++)
		    atomicAdd(&(res[row+y]), temp[y][0]);
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
    long N = blockSize * blockSize; //(1UL<<10); // 2^10

    double* x = (double*) malloc(N * N * sizeof(double));
    double* y = (double*) malloc(N * sizeof(double));
    double* res = (double*) malloc(N * sizeof(double));
    double* res_ref = (double*) malloc(N * sizeof(double));

    for (long i = 0; i < N; i++) {
        y[i] = 1.0/(i+1);
        res[i] = 0;
        res_ref[i] = 0;
        for(long j = 0; j < N; j++){
            x[i * N + j] = i+1;
        }
    }

    double tt = omp_get_wtime();
    matrix_vec_mul(res_ref, x, y, N);
    printf("CPU Bandwidth = %f GB/s\n", (N * N + N)*sizeof(double) / (omp_get_wtime()-tt)/1e9);

    double *x_d, *y_d, *res_d;
    cudaMalloc(&x_d, N * N*sizeof(double));
    Check_CUDA_Error("malloc x failed");
    cudaMalloc(&y_d, N*sizeof(double));
    cudaMalloc(&res_d, N*sizeof(double));
    Check_CUDA_Error("memset failed");

    dim3 grid(blockSize, blockSize);
    dim3 block(blockSize, blockSize);

    tt = omp_get_wtime();
    cudaMemcpy(x_d, x, N * N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(res_d, res, sizeof(double), cudaMemcpyHostToDevice);
    matrix_vec_product_kernel<<<grid, block>>>(res_d, x_d, y_d, N);
    cudaDeviceSynchronize();
    cudaMemcpy(res, res_d, N*sizeof(double), cudaMemcpyDeviceToHost);
    printf("GPU Bandwidth = %f GB/s\n", (N * N + N)*sizeof(double) / (omp_get_wtime()-tt)/1e9);

    double err = 0;
    for(long i = 0; i < N; i++) {//printf("ref[i]:%f, res[i]:%f\n", res_ref[i], res[i]);
	err += res_ref[i]-res[i];}
    printf("Error = %f\n", err);


    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(res_d);

    free(x);
    free(y);
    free(res);
    free(res_ref);

    return 0;
  }
