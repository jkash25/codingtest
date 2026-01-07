NVCC = nvcc
CXXFLAGS = -O3 -std=c++17
LDFLAGS = -lcublas

TARGET = geglu_ffn
SRCS = ffn.cpp ffn.cu
ARCH = -arch=sm_86
all:
	$(NVCC) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(ARCH) $(LDFLAGS)

clean:
	rm -f $(TARGET)
