#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>
#include <cmath>

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

const int HIDDEN = 4096;
const int INTER = 12288;

cublasLtHandle_t ltHandle;
cublasLtMatmulPreference_t preference;
bool initialized = false;
bool algos_found = false;

//gpu pointers
__half *W_combined = nullptr;
__half *Wo = nullptr;
__half *d_u = nullptr;
__half *d_h = nullptr;
void* workspace = nullptr;
size_t workspaceSize = 1024 * 1024 * 32; //32 mb workspace

cublasLtMatmulAlgo_t algo_gemm1;
cublasLtMatmulAlgo_t algo_gemm2;


__global__ void geglu_kernel(
    const __half2* __restrict__ input, 
    __half2* __restrict__ output, 
    int iter_size_h2, 
    int batch_stride_h2) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;
    
    if (idx < iter_size_h2) {
        int base_offset = batch_idx * batch_stride_h2;
        __half2 val = input[base_offset + idx];
        __half2 gate = input[base_offset + iter_size_h2 + idx];
        
        //approximate gelu x * sigmoid(1.702 * x)
        const __half2 k1 = __float2half2_rn(1.702f);
        const __half2 kOne = __float2half2_rn(1.0f);
        
        __half2 g_scaled = __hmul2(gate, k1);
        __half2 neg_g = __hneg2(g_scaled);
        __half2 exp_val = h2exp(neg_g);
        __half2 denom = __hadd2(kOne, exp_val);
        __half2 sigmoid_g = h2rcp(denom);
        
        //geglu: val * sigmoid(gate)
        output[batch_idx * iter_size_h2 + idx] = __hmul2(val, sigmoid_g);
    }
}

//helper to init weights on device
__global__ void init_kernel(__half* data, size_t size, float seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = __float2half(sinf(idx + seed) * 0.01f);
    }
}

void random_init(__half* data, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    init_kernel<<<blocks, threads>>>(data, size, (float)rand());
}

void find_best_algo(cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, const void* alpha, const void* beta, const void* A, const void* B, void* C, cublasLtMatmulAlgo_t& bestAlgo) {
    int returnedResults = 0;
    const int max_algos = 10;
    cublasLtMatmulHeuristicResult_t heuristicResults[max_algos];

    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, 
        preference, max_algos, heuristicResults, &returnedResults
    ));

    if (returnedResults == 0) {
        fprintf(stderr, "No cuBLASLt algorithm found!\n");
        exit(1);
    }
    bestAlgo = heuristicResults[0].algo;
}

void initialize_once() {
    if (initialized) return;
    
    CUBLAS_CHECK(cublasLtCreate(&ltHandle));
    CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
        &workspaceSize, sizeof(workspaceSize)));

    CUDA_CHECK(cudaMalloc(&W_combined, HIDDEN * (2 * INTER) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&Wo, (INTER) * HIDDEN * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_u, 128 * (2 * INTER) * sizeof(__half))); 
    CUDA_CHECK(cudaMalloc(&d_h, 128 * INTER * sizeof(__half)));
    
    initialized = true;
}

void geglu(const __half* x, __half* out, int B) {
    if (!initialized) {
        initialize_once();
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    //gemm 1: (b, hidden) @ (hidden, 2*inter)
    cublasLtMatmulDesc_t opDesc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t opN = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    cublasLtMatrixLayout_t xDesc, wDesc, uDesc;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&xDesc, CUDA_R_16F, HIDDEN, B, HIDDEN));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&wDesc, CUDA_R_16F, 2 * INTER, HIDDEN, 2 * INTER));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&uDesc, CUDA_R_16F, 2 * INTER, B, 2 * INTER));

    if (!algos_found) {
        find_best_algo(opDesc, wDesc, xDesc, uDesc, &alpha, &beta, W_combined, x, d_u, algo_gemm1);
    }

    CUBLAS_CHECK(cublasLtMatmul(
        ltHandle, opDesc, &alpha, W_combined, wDesc, x, xDesc, &beta, d_u, uDesc, d_u, uDesc, 
        &algo_gemm1, workspace, workspaceSize, 0));

    //
    int iter_size_h2 = INTER / 2;
    int batch_stride_h2 = (2 * INTER) / 2;
    dim3 threads(256);
    dim3 blocks((iter_size_h2 + threads.x - 1) / threads.x, B);

    geglu_kernel<<<blocks, threads>>>(
        (const __half2*)d_u, (__half2*)d_h, iter_size_h2, batch_stride_h2
    );

    //gemm 2: (B, Inter) @ (inter, hidden)
    cublasLtMatrixLayout_t hDesc, woDesc, outDesc;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&hDesc, CUDA_R_16F, INTER, B, INTER));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&woDesc, CUDA_R_16F, HIDDEN, INTER, HIDDEN));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&outDesc, CUDA_R_16F, HIDDEN, B, HIDDEN));

    if (!algos_found) {
        find_best_algo(opDesc, woDesc, hDesc, outDesc, &alpha, &beta, Wo, d_h, out, algo_gemm2);
        algos_found = true; 
    }

    CUBLAS_CHECK(cublasLtMatmul(
        ltHandle, opDesc, &alpha, Wo, woDesc, d_h, hDesc, &beta, out, outDesc, out, outDesc, 
        &algo_gemm2, workspace, workspaceSize, 0
    ));

    // Cleanup Descriptors
    cublasLtMatmulDescDestroy(opDesc);
    cublasLtMatrixLayoutDestroy(xDesc); 
    cublasLtMatrixLayoutDestroy(wDesc); 
    cublasLtMatrixLayoutDestroy(uDesc);
    cublasLtMatrixLayoutDestroy(hDesc); 
    cublasLtMatrixLayoutDestroy(woDesc); 
    cublasLtMatrixLayoutDestroy(outDesc);
}

void cleanup() {
    CUDA_CHECK(cudaFree(W_combined));
    CUDA_CHECK(cudaFree(Wo));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_h));
    CUDA_CHECK(cudaFree(workspace));
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLAS_CHECK(cublasLtDestroy(ltHandle));
}