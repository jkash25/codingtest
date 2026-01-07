// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cublas_v2.h>
// #include <math.h>
// #include <random>
// #define CUDA_CHECK(call) \
//     do { \
//         cudaError_t error = call; \
//         if (error != cudaSuccess) { \
//             fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__, __LINE__, \
//                     cudaGetErrorString(error), error); \
//             cudaDeviceReset(); \
//             exit(EXIT_FAILURE); \
//         } \
//     } while(0)

// #define CUBLAS_CHECK(call) \
//     do { \
//         cublasStatus_t status = call; \
//         if (status != CUBLAS_STATUS_SUCCESS) { \
//             fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
//             exit(EXIT_FAILURE); \
//         } \
//     } while(0)


// const int HIDDEN = 4096;
// const int INTER = 12288;

// static cublasHandle_t handle;
// static bool initialized = false;

// float* Wu = nullptr;
// float *Wv = nullptr;
// float *Wo = nullptr;

// float *d_u = nullptr;
// float *d_v = nullptr;
// float *d_h = nullptr;

// __device__ __forceinline__ float gelu(float x) {
//     //tanh approximation used by pytorch
//     const float c = 0.7978845608f;  //sqrt(2/pi)
//     const float c1  = 0.044715f;
//     float x3 = x * x * x;
//     return 0.5f * x * (1.0f + tanhf(c * (x + c1 * x3)));
// }

// __global__ void geglu_kernel(const float* u, const float* v, float* h,int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n) {
//         //float u_val = u[idx];
//         //float v_val = v[idx];
//         h[idx] = gelu(u[idx]) * v[idx];
//     }
// }

// void initialize_once() {
//     if (initialized) return;

//     CUBLAS_CHECK(cublasCreate(&handle));
//     CUDA_CHECK(cudaMalloc(&Wu, HIDDEN * INTER * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&Wv, HIDDEN * INTER * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&Wo, INTER * HIDDEN * sizeof(float)));

//     const int MAX_B = 128;
//     CUDA_CHECK(cudaMalloc(&d_u, MAX_B * INTER * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_v, MAX_B * INTER * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_h, MAX_B * INTER * sizeof(float)));

//     size_t size_u_v = HIDDEN * INTER;
//     size_t size_o = INTER * HIDDEN;

//     float *h_Wu = (float*)malloc(size_u_v * sizeof(float));
//     float *h_Wv = (float*)malloc(size_u_v * sizeof(float));
//     float *h_Wo = (float*)malloc(size_o * sizeof(float));

//     std::default_random_engine generator(42); //fix seed
//     float std_dev = sqrt(1.0f / HIDDEN);
//     std::normal_distribution<float> distribution(0.0f, std_dev);

//     for (size_t i = 0; i < size_u_v; i++) {
//         h_Wu[i] = distribution(generator);
//         h_Wv[i] = distribution(generator);
//     }
//     for (size_t i = 0; i < size_o; i++) {
//         h_Wo[i] = distribution(generator);
//     }

//     CUDA_CHECK(cudaMemcpy(Wu, h_Wu, size_u_v * sizeof(float), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(Wv, h_Wv, size_u_v * sizeof(float), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(Wo, h_Wo, size_o * sizeof(float), cudaMemcpyHostToDevice));

//     free(h_Wu);
//     free(h_Wv);
//     free(h_Wo);
//     initialized = true;
// }

// void geglu(const float* x, float* out, int B) {
//     initialize_once();
//     float alpha = 1.0f;
//     float beta = 0.0f;

//     float *u, *v, *h;
//     CUDA_CHECK(cudaMalloc(&u, B * INTER * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&v, B * INTER * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&h, B * INTER * sizeof(float)));

//     // u = x * Wu
//     CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, INTER, B, HIDDEN, &alpha, Wu, INTER, x, HIDDEN, &beta, u, INTER));

//     // v = x * Wv
//     CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, INTER, B, HIDDEN, &alpha,Wv, INTER, x, HIDDEN, &beta, v, INTER));
    
//     // h = geglu(u, v)
//     int n = B * INTER;
//     int threads = 256;
//     int blocks = (n + threads - 1) / threads;
//     geglu_kernel<<<blocks, threads>>>(u, v, h, n);
//     CUDA_CHECK(cudaGetLastError());

//     // out = h * Wo
//     CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HIDDEN, B, INTER, &alpha, Wo, HIDDEN, h, INTER, &beta, out, HIDDEN));

//     // CUDA_CHECK(cudaFree(u));
//     // CUDA_CHECK(cudaFree(v));
//     // CUDA_CHECK(cudaFree(h));
// }

//==============================
// Trial 2
//Results
//GEGLU FFN BENCHMARK
// Batch Size: 4
// Average Time over 50 runs: 3.84265 ms
// Batch Size: 8
// Average Time over 50 runs: 3.94589 ms
// Batch Size: 16
// Average Time over 50 runs: 4.12418 ms
// Batch Size: 32
// Average Time over 50 runs: 4.11694 ms
// Batch Size: 64
// Average Time over 50 runs: 7.50258 ms
// Batch Size: 128
// Average Time over 50 runs: 10.1844 ms
//==============================

// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cublas_v2.h>
// #include <math.h>
// #include <random>
// #include <stdio.h>

// #define CUDA_CHECK(call) \
//     do { \
//         cudaError_t error = call; \
//         if (error != cudaSuccess) { \
//             fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__, __LINE__, \
//                     cudaGetErrorString(error), error); \
//             cudaDeviceReset(); \
//             exit(EXIT_FAILURE); \
//         } \
//     } while(0)

// #define CUBLAS_CHECK(call) \
//     do { \
//         cublasStatus_t status = call; \
//         if (status != CUBLAS_STATUS_SUCCESS) { \
//             fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
//             exit(EXIT_FAILURE); \
//         } \
//     } while(0)

// const int HIDDEN = 4096;
// const int INTER = 12288;

// static cublasHandle_t handle;
// static bool initialized = false;

// float* Wu = nullptr;
// float* Wv = nullptr;
// float* Wo = nullptr;

// // Preallocated buffers
// float* d_u = nullptr;
// float* d_v = nullptr;
// float* d_h = nullptr;

// __device__ __forceinline__ float gelu(float x) {
//     const float c = 0.7978845608f;  // sqrt(2/pi)
//     const float c1 = 0.044715f;
//     float x3 = x * x * x;
//     return 0.5f * x * (1.0f + tanhf(c * (x + c1 * x3)));
// }

// // Fused GELU + multiply kernel
// __global__ void geglu_kernel(const float* u, const float* v, float* h, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n) {
//         h[idx] = gelu(u[idx]) * v[idx];
//     }
// }

// // Initialize weights and preallocate buffers once
// void initialize_once() {
//     if (initialized) return;

//     CUBLAS_CHECK(cublasCreate(&handle));

//     CUDA_CHECK(cudaMalloc(&Wu, HIDDEN * INTER * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&Wv, HIDDEN * INTER * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&Wo, INTER * HIDDEN * sizeof(float)));

//     const int MAX_B = 128;  // Max batch size
//     CUDA_CHECK(cudaMalloc(&d_u, MAX_B * INTER * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_v, MAX_B * INTER * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_h, MAX_B * INTER * sizeof(float)));

//     // Initialize host weights
//     size_t size_u_v = HIDDEN * INTER;
//     size_t size_o = INTER * HIDDEN;

//     float* h_Wu = (float*)malloc(size_u_v * sizeof(float));
//     float* h_Wv = (float*)malloc(size_u_v * sizeof(float));
//     float* h_Wo = (float*)malloc(size_o * sizeof(float));

//     std::default_random_engine generator(42);
//     float std_dev = sqrt(1.0f / HIDDEN);
//     std::normal_distribution<float> distribution(0.0f, std_dev);

//     for (size_t i = 0; i < size_u_v; i++) {
//         h_Wu[i] = distribution(generator);
//         h_Wv[i] = distribution(generator);
//     }
//     for (size_t i = 0; i < size_o; i++) {
//         h_Wo[i] = distribution(generator);
//     }

//     CUDA_CHECK(cudaMemcpy(Wu, h_Wu, size_u_v * sizeof(float), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(Wv, h_Wv, size_u_v * sizeof(float), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(Wo, h_Wo, size_o * sizeof(float), cudaMemcpyHostToDevice));

//     free(h_Wu);
//     free(h_Wv);
//     free(h_Wo);

//     initialized = true;
// }

// // Optimized GEGLU FFN
// void geglu(const float* x, float* out, int B) {
//     initialize_once();

//     float alpha = 1.0f;
//     float beta = 0.0f;

//     // 1) Compute u = x @ Wu
//     CUBLAS_CHECK(cublasSgemm(handle,
//                              CUBLAS_OP_N, CUBLAS_OP_N,
//                              INTER, B, HIDDEN,
//                              &alpha,
//                              Wu, INTER,
//                              x, HIDDEN,
//                              &beta,
//                              d_u, INTER));

//     // 2) Compute v = x @ Wv
//     CUBLAS_CHECK(cublasSgemm(handle,
//                              CUBLAS_OP_N, CUBLAS_OP_N,
//                              INTER, B, HIDDEN,
//                              &alpha,
//                              Wv, INTER,
//                              x, HIDDEN,
//                              &beta,
//                              d_v, INTER));

//     // 3) Apply GELU(u) * v → h
//     int n = B * INTER;
//     int threads = 256;
//     int blocks = (n + threads - 1) / threads;
//     geglu_kernel<<<blocks, threads>>>(d_u, d_v, d_h, n);
//     CUDA_CHECK(cudaGetLastError());

//     // 4) Final projection: out = h @ Wo
//     CUBLAS_CHECK(cublasSgemm(handle,
//                              CUBLAS_OP_N, CUBLAS_OP_N,
//                              HIDDEN, B, INTER,
//                              &alpha,
//                              Wo, HIDDEN,
//                              d_h, INTER,
//                              &beta,
//                              out, HIDDEN));
// }

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math.h>
#include <random>
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error), error); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

const int HIDDEN = 4096;
const int INTER = 12288;

static cublasHandle_t handle;
static bool initialized = false;

__half *Wu = nullptr;
__half *Wv = nullptr;
__half *Wo = nullptr;

// Preallocated buffers for max batch size 128
__half *d_u = nullptr;
__half *d_v = nullptr;
__half *d_h = nullptr;

// Half-precision GELU using Tensor Core friendly approximation
__device__ __forceinline__ __half gelu_half(__half x) {
    // Tanh approximation
    float xf = __half2float(x);
    const float c = 0.7978845608f;
    const float c1 = 0.044715f;
    float x3 = xf * xf * xf;
    float y = 0.5f * xf * (1.0f + tanhf(c * (xf + c1 * x3)));
    return __float2half(y);
}

// Fused GELU(u) * v kernel in half precision
__global__ void geglu_half_kernel(const __half* u, const __half* v, __half* h, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        h[idx] = __hmul(gelu_half(u[idx]), v[idx]);
    }
}

// Initialize weights and buffers
void initialize_once() {
    if (initialized) return;

    CUBLAS_CHECK(cublasCreate(&handle));

    CUDA_CHECK(cudaMalloc(&Wu, HIDDEN * INTER * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&Wv, HIDDEN * INTER * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&Wo, INTER * HIDDEN * sizeof(__half)));

    const int MAX_B = 128;
    CUDA_CHECK(cudaMalloc(&d_u, MAX_B * INTER * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_v, MAX_B * INTER * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_h, MAX_B * INTER * sizeof(__half)));

    size_t size_u_v = HIDDEN * INTER;
    size_t size_o = INTER * HIDDEN;

    std::vector<float> h_Wu(size_u_v);
    std::vector<float> h_Wv(size_u_v);
    std::vector<float> h_Wo(size_o);

    std::default_random_engine generator(42);
    float std_dev = sqrt(1.0f / HIDDEN);
    std::normal_distribution<float> dist(0.0f, std_dev);

    for (size_t i = 0; i < size_u_v; i++) {
        h_Wu[i] = dist(generator);
        h_Wv[i] = dist(generator);
    }
    for (size_t i = 0; i < size_o; i++) {
        h_Wo[i] = dist(generator);
    }

    // Convert to __half on device
    std::vector<__half> hWu_half(size_u_v);
    std::vector<__half> hWv_half(size_u_v);
    std::vector<__half> hWo_half(size_o);

    for (size_t i = 0; i < size_u_v; i++) {
        hWu_half[i] = __float2half(h_Wu[i]);
        hWv_half[i] = __float2half(h_Wv[i]);
    }
    for (size_t i = 0; i < size_o; i++) {
        hWo_half[i] = __float2half(h_Wo[i]);
    }

    CUDA_CHECK(cudaMemcpy(Wu, hWu_half.data(), size_u_v * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Wv, hWv_half.data(), size_u_v * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Wo, hWo_half.data(), size_o * sizeof(__half), cudaMemcpyHostToDevice));

    initialized = true;
}

// Optimized GEGLU FFN in half precision using Tensor Cores
void geglu(const __half* x, __half* out, int B) {
    initialize_once();

    // GEMM params for half precision with Tensor Cores
    float alpha_f = 1.0f, beta_f = 0.0f;

    // Compute u = x @ Wu
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              INTER, B, HIDDEN,
                              &alpha_f,
                              Wu, CUDA_R_16F, INTER,
                              x, CUDA_R_16F, HIDDEN,
                              &beta_f,
                              d_u, CUDA_R_16F, INTER,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Compute v = x @ Wv
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              INTER, B, HIDDEN,
                              &alpha_f,
                              Wv, CUDA_R_16F, INTER,
                              x, CUDA_R_16F, HIDDEN,
                              &beta_f,
                              d_v, CUDA_R_16F, INTER,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Fused GELU(u) * v
    int n = B * INTER;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    geglu_half_kernel<<<blocks, threads>>>(d_u, d_v, d_h, n);
    CUDA_CHECK(cudaGetLastError());

    // Final projection: out = h @ Wo using Tensor Cores
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              HIDDEN, B, INTER,
                              &alpha_f,
                              Wo, CUDA_R_16F, HIDDEN,
                              d_h, CUDA_R_16F, INTER,
                              &beta_f,
                              out, CUDA_R_16F, HIDDEN,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
