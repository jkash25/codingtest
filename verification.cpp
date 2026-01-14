#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <iostream>
#include "ffn.h"

//cpu gelu approximation
float cpu_gelu(float x) {
    return x * (1.0f / (1.0f + expf(-1.702f * x)));
}

int main() {
    const int HIDDEN = 4096;
    const int INTER = 12288;
    initialize_once();
    
    int B = 4; //change batch size to verify all sizes
    size_t x_size = B * HIDDEN;
    size_t out_size = B * HIDDEN;

    
    std::vector<__half> h_x(x_size);
    std::vector<__half> h_out_gpu(out_size);
    std::vector<__half> h_W1(HIDDEN * 2 * INTER);
    std::vector<__half> h_W2(INTER * HIDDEN);

    __half *d_x, *d_out;
    cudaMalloc(&d_x, x_size * sizeof(__half));
    cudaMalloc(&d_out, out_size * sizeof(__half));

    random_init(W_combined, HIDDEN * 2 * INTER);
    random_init(Wo, INTER * HIDDEN);
    random_init(d_x, x_size);

    cudaMemcpy(h_W1.data(), W_combined, HIDDEN * 2 * INTER * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_W2.data(), Wo, INTER * HIDDEN * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x.data(), d_x, x_size * sizeof(__half), cudaMemcpyDeviceToHost);

    geglu(d_x, d_out, B);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out_gpu.data(), d_out, out_size * sizeof(__half), cudaMemcpyDeviceToHost);

    printf("Computing CPU reference...\n");
    std::vector<float> cpu_intermediate(B * 2 * INTER);
    std::vector<float> cpu_after_geglu(B * INTER);
    std::vector<float> cpu_final(B * HIDDEN, 0.0f);

    //gemm 1
    for (int b = 0; b < B; ++b) {
        for (int j = 0; j < 2 * INTER; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < HIDDEN; ++k) {
                sum += __half2float(h_x[b * HIDDEN + k]) * __half2float(h_W1[k * (2 * INTER) + j]);
            }
            cpu_intermediate[b * (2 * INTER) + j] = sum;
        }
    }

    //geglu activation
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < INTER; ++i) {
            float val = cpu_intermediate[b * (2 * INTER) + i];
            float gate = cpu_intermediate[b * (2 * INTER) + INTER + i];
            cpu_after_geglu[b * INTER + i] = val * cpu_gelu(gate);
        }
    }

    // 3.gemm 2
    for (int b = 0; b < B; ++b) {
        for (int j = 0; j < HIDDEN; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < INTER; ++k) {
                sum += cpu_after_geglu[b * INTER + k] * __half2float(h_W2[k * HIDDEN + j]);
            }
            cpu_final[b * HIDDEN + j] = sum;
        }
    }

    double mse = 0;
    float max_error = 0;
    for (int i = 0; i < out_size; ++i) {
        float gpu_val = __half2float(h_out_gpu[i]);
        float cpu_val = cpu_final[i];
        float error = std::abs(gpu_val - cpu_val);
        mse += error * error;
        if (error > max_error) max_error = error;
    }
    mse /= out_size;

    printf("\nVERIFICATION RESULTS\n");
    printf("--------------------------\n");
    printf("Mean Squared Error (MSE): %e\n", mse);
    printf("Max Absolute Error: %f\n", max_error);

    if (mse < 1e-4) {
        printf("RESULT: PASS\n");
    } else {
        printf("RESULT: FAIL\n");
    }
    cudaFree(d_x);
    cudaFree(d_out);
    return 0;
}