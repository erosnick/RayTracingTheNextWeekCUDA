
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include <cstdint>
#include <cstdio>

__global__ void kernel(int32_t* counter) {
    auto old = atomicAdd(counter, 1);
    //__syncthreads();
    if ((old + 1) > 0 && (old + 1) % 64 == 0) {
        printf("%d, %.2f\n", old + 1, (float((old + 1) * 100) / 4096));
    }
}

int main() {
    constexpr auto width = 320;
    constexpr auto height = 320;

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    int32_t* counter = nullptr;
    cudaMallocManaged(&counter, sizeof(int32_t*));

    //kernel<<<gridSize, blockSize>>>(counter);
    kernel<<<64, 64>>>(counter);
    cudaDeviceSynchronize();

    cudaFree(counter);

    return 0;
}
