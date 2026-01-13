# NVCC = nvcc
# CXXFLAGS = -O3 -std=c++17
# LDFLAGS = -lcublas -lcublasLt

# TARGET = run_benchmark
# SRCS = benchmark.cpp ffn.cu
# ARCH = -arch=sm_86
# all:
# 	$(NVCC) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(ARCH) $(LDFLAGS)

# clean:
# 	rm -f $(TARGET)
NVCC = nvcc
CXXFLAGS = -O3 -std=c++17
LDFLAGS = -lcublas -lcublasLt
ARCH = -arch=sm_86

# Output binary names
BENCHMARK_TARGET = run_benchmark
VERIFY_TARGET = run_verify

# Source files
KERNEL_SRC = ffn.cu
BENCHMARK_SRC = benchmark.cpp
VERIFY_SRC = verification.cpp

# The 'all' target will now build both binaries
all: $(BENCHMARK_TARGET) $(VERIFY_TARGET)

# Rule for building the benchmark
$(BENCHMARK_TARGET): $(BENCHMARK_SRC) $(KERNEL_SRC)
	$(NVCC) $(CXXFLAGS) $(BENCHMARK_SRC) $(KERNEL_SRC) -o $(BENCHMARK_TARGET) $(ARCH) $(LDFLAGS)

# Rule for building the verification
$(VERIFY_TARGET): $(VERIFY_SRC) $(KERNEL_SRC)
	$(NVCC) $(CXXFLAGS) $(VERIFY_SRC) $(KERNEL_SRC) -o $(VERIFY_TARGET) $(ARCH) $(LDFLAGS)

clean:
	rm -f $(BENCHMARK_TARGET) $(VERIFY_TARGET)