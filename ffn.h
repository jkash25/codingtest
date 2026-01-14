#ifndef FFN_H
#define FFN_H

#include <cuda_fp16.h>
#include <cublasLt.h>

extern const int HIDDEN;
extern const int INTER;

extern __half *W_combined;
extern __half *Wo;
extern __half *d_u;
extern __half *d_h;
extern void* workspace;
extern cublasLtHandle_t ltHandle;
extern cublasLtMatmulPreference_t preference;

void geglu(const __half* x, __half* out, int B);
void initialize_once();
void random_init(__half* data, size_t size);

#endif