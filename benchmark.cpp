#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include "ffn.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error), error); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int)status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    initialize_once();
    int warmup = 5;
    int batch_sizes[] = {4, 8, 16, 32, 64, 128};
    const int NUM_RUNS = 50;
    const int HIDDEN = 4096;
    const int INTER = 12288;
    __half *d_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, 128 * HIDDEN * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, 128 * HIDDEN * sizeof(__half)));

    random_init(W_combined, HIDDEN * (2 * INTER));
    random_init(Wo, INTER * HIDDEN);

    printf("GEGLU FFN BENCHMARK\n");
    printf("----------------------------------\n");

    for (int B : batch_sizes) {
        random_init(d_x, B * HIDDEN);

        //warmup
        for(int i = 0; i < warmup; i++) {
            geglu(d_x, d_out, B);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < NUM_RUNS; i++) {
            geglu(d_x, d_out, B);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("Batch Size: %3d | Avg Time: %8.5f ms\n", B, ms / NUM_RUNS);

        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(W_combined)); CUDA_CHECK(cudaFree(Wo));
    CUDA_CHECK(cudaFree(d_u)); CUDA_CHECK(cudaFree(d_h));
    CUDA_CHECK(cudaFree(workspace));
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLAS_CHECK(cublasLtDestroy(ltHandle));
    
    return 0;
}