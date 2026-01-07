#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "ffn.h"
#include <random>
#include <chrono>
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

void initialize(float* data, int n) {
    std::default_random_engine gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n; ++i) {
        data[i] = dist(gen);
    }
}

bool verify_results(float* h_out, float* h_x, float* h_Wu, float* h_Wv, float* h_Wo, int B, int hidden, int inter) {
    printf("Verifying Sample of Batch %d... ", B);
    float epsilon = 1e-3f; 

    // Just verify the first row (batch 0) and first few columns to save time
    for (int b = 0; b < 1; ++b) {
        for (int j = 0; j < 10; ++j) { // Check first 10 output elements
            float expected_val = 0.0f;

            for (int k = 0; k < inter; ++k) {
                float u = 0.0f;
                float v = 0.0f;
                for (int i = 0; i < hidden; ++i) {
                    u += h_x[b * hidden + i] * h_Wu[i * inter + k];
                    v += h_x[b * hidden + i] * h_Wv[i * inter + k];
                }

                float gelu_u = 0.5f * u * (1.0f + tanhf(0.7978845608f * (u + 0.044715f * u * u * u)));
                float h_element = gelu_u * v;

                expected_val += h_element * h_Wo[k * hidden + j];
            }

            float actual_val = h_out[b * hidden + j];
            if (std::abs(expected_val - actual_val) > epsilon) {
                printf("FAILED at [%d, %d]: Expected %f, Got %f\n", b, j, expected_val, actual_val);
                return false;
            }
        }
    }
    printf("SAMPLE PASSED!\n");
    return true;
}

int main() {
    const int HIDDEN = 4096;
    const int INTER = 12288;
    const int NUM_TRIALS = 50;
    const int NUM_WARMUP = 10;
    std::vector<int> batch_sizes = {4, 8, 16, 32, 64, 128};

    std::cout << "GEGLU FFN BENCHMARK" << std::endl;

    //warmup
    initialize_once();

    for (int B : batch_sizes) {
        std::cout << "Batch Size: " << B << std::endl;

        float *x, *out;
        CUDA_CHECK(cudaMalloc(&x,  B * HIDDEN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&out,B * HIDDEN * sizeof(float)));

        std::vector<float> h_x(B * HIDDEN);
        initialize(h_x.data(), h_x.size());

        CUDA_CHECK(cudaMemcpy(x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));

        //warmup
        for (int i = 0; i < NUM_WARMUP; i++) {
            geglu(x, out, B);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // if (B <= 8) { // Only verify small batches to save time
        //     std::vector<float> h_out(B * HIDDEN);
        //     std::vector<float> h_Wu_cpu(HIDDEN * INTER);
        //     std::vector<float> h_Wv_cpu(HIDDEN * INTER);
        //     std::vector<float> h_Wo_cpu(INTER * HIDDEN);

        //     // Copy back everything for CPU verification
        //     CUDA_CHECK(cudaMemcpy(h_out.data(), out, B * HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));
        //     CUDA_CHECK(cudaMemcpy(h_Wu_cpu.data(), Wu, HIDDEN * INTER * sizeof(float), cudaMemcpyDeviceToHost));
        //     CUDA_CHECK(cudaMemcpy(h_Wv_cpu.data(), Wv, HIDDEN * INTER * sizeof(float), cudaMemcpyDeviceToHost));
        //     CUDA_CHECK(cudaMemcpy(h_Wo_cpu.data(), Wo, INTER * HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));

        //     if (!verify_results(h_out.data(), h_x.data(), h_Wu_cpu.data(), h_Wv_cpu.data(), h_Wo_cpu.data(), B, HIDDEN, INTER)) {
        //         exit(1); 
        //     }
        // }

        double total_ms = 0.0;
        for (int i = 0; i < NUM_TRIALS; i++) {
            CUDA_CHECK(cudaDeviceSynchronize());
            auto start = std::chrono::high_resolution_clock::now();
            geglu(x, out, B);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            total_ms += duration.count();
        }
        double avg_ms = total_ms / NUM_TRIALS;
        std::cout << "Average Time over " << NUM_TRIALS << " runs: " << avg_ms << " ms" << std::endl;
        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaFree(out));
    }
    return 0;
}
