#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CUDATypes.h"
#include "Utils.h"

#include <cstdint>
#include <cassert>

template<typename T>
class Buffer {
public:
    Buffer() {}
    Buffer(int32_t inSize) {
        initialize(inSize);
    }

    ~Buffer() {
        uninitialize();
    }

    void initialize(int32_t inSize) {
        size = inSize;
        gpuErrorCheck(cudaMallocManaged(&buffer, sizeof(T) * size));
    }

    void uninitialize() {
        gpuErrorCheck(cudaFree(buffer));
    }

    CUDA_HOST_DEVICE T operator[](int32_t index) const {
        if (index >= size)
        {
            printf("Index out of range.\n");
            assert(index < size);
        }

        return buffer[index];
    }

    CUDA_HOST_DEVICE T& operator[](int32_t index) {
        if (index >= size)
        {
            printf("Index out of range.\n");
            assert(index < size);
        }
        return buffer[index];
    }

    T* get() {
        return buffer;
    }

    T* buffer = nullptr;
    int32_t size = 0;
};