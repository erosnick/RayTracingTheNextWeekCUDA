#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include "CUDATypes.h"

#include <string>
#include <algorithm>
#include <random>
#include <iostream>

#define gpuErrorCheck(ans) { Utils::gpuAssert((ans), __FILE__, __LINE__); }

namespace Utils {
    //inline float randomFloat(float start = 0.0f, float end = 1.0f) {
    //    std::uniform_real_distribution<float> distribution(start, end);
    //    static std::random_device randomDevice;
    //    static std::mt19937 generator(randomDevice());
    //    return distribution(generator);
    //}

    inline float randomFloat() {
        static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        static std::random_device randomDevice;
        static std::mt19937 generator(randomDevice());
        return distribution(generator);
    }

    inline float randomFloat(Float min, Float max) {
        return (min + (max - min) * randomFloat());
    }

    inline double randomDouble(double start = 0.0, double end = 1.0) {
        std::uniform_real_distribution<double> distribution(start, end);
        static std::random_device randomDevice;
        static std::mt19937 generator(randomDevice());
        return distribution(generator);
    }

    // Random double in [0, 1]
    CUDA_DEVICE inline Float random(curandState* randState) {
        return curand_uniform(randState);
    }

    // Random double in [min, max]
    CUDA_DEVICE inline Float random(curandState* randState, Float min, Float max) {
        return min + (max - min) * random(randState);
    }


    // Random float3 in [0, 1]
    CUDA_DEVICE inline Float3 randomVector(curandState* randState) {
        Float x = curand_uniform(randState);
        Float y = curand_uniform(randState);
        Float z = curand_uniform(randState);

        return make_float3(x, y, z);
    }

    // Random float3 in [min, max]
    CUDA_DEVICE inline Float3 randomVector(curandState* randState, Float min, Float max) {
        return (min + (max - min) * randomVector(randState));
    }

    // Random float3 in unit sphere
    CUDA_DEVICE inline Float3 randomInUnitSphere(curandState* randState) {
        while (true) {
            auto position = randomVector(randState, -1.0f, 1.0f);
            if (lengthSquared(position) >= 1.0f) {
                continue;
            }

            return position;
        }
    }

    // Random float3 in unit sphere surface
    CUDA_DEVICE inline Float3 randomUnitVector(curandState* randState) {
        return normalize(randomInUnitSphere(randState));
    }

    // Random float3 in hemisphere
    CUDA_DEVICE inline Float3 randomHemiSphere(const Float3& normal, curandState* randState) {
        auto inUnitSphere = randomInUnitSphere(randState);

        //  In the same hemisphere as the normal
        if (dot(inUnitSphere, normal) > 0.0f) {
            return inUnitSphere;
        }
        return -inUnitSphere;
    }

    CUDA_DEVICE inline Float3 randomInUnitDisk(curandState* randState) {
        while (true) {
            auto position = make_float3(random(randState, -1.0f, 1.0f), random(randState, -1.0f, 1.0f), 0.0f);
            if (lengthSquared(position) >= 1.0f) {
                continue;
            }
            return position;
        }
    }

    CUDA_DEVICE inline bool nearZero(const Float3& v) {
        // Return true if the vector is close to zero in all dimensions.
        const auto s = 1e-8;
        return (fabs(v.x) < s) && (fabs(v.y) < s) && (fabs(v.z) < s);
    }

    CUDA_DEVICE inline Float3 reflect(const Float3& v, const Float3& n) {
        return v - 2.0f * dot(v, n) * n;
    }

    CUDA_DEVICE inline Float3 refract(const Float3& uv, const Float3& n, Float etaiOverEtat) {
        auto cosTheta = fmin(dot(-uv, n), 1.0f);
        auto rOutPerp = etaiOverEtat * (uv + cosTheta * n);
        auto rOutParallel = -sqrt(fabs(1.0f - lengthSquared(rOutPerp))) * n;
        return rOutPerp + rOutParallel;
    }

    void openImage(const std::wstring& path);

    inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
        if (code != cudaSuccess)
        {
            fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            // Make sure we call CUDA Device Reset before exiting
            cudaDeviceReset();
            if (abort) exit(code);
        }
    }

    inline void reportGPUUsageInfo() {
        size_t freeBytes;
        size_t totalBytes;

        gpuErrorCheck(cudaMemGetInfo(&freeBytes, &totalBytes));

        auto freeDb = (double)freeBytes;

        auto totalDb = (double)totalBytes;

        auto usedDb = totalDb - freeDb;

        printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
            usedDb / 1024.0 / 1024.0, freeDb / 1024.0 / 1024.0, totalDb / 1024.0 / 1024.0);
    }

    inline void queryDeviceProperties() {
        int32_t deviceIndex = 0;
        cudaDeviceProp devicePro;
        cudaGetDeviceProperties(&devicePro, deviceIndex);

        std::cout << "使用的GPU device：" << deviceIndex << ": " << devicePro.name << std::endl;
        std::cout << "SM的数量：" << devicePro.multiProcessorCount << std::endl;
        std::cout << "每个线程块的共享内存大小：" << devicePro.sharedMemPerBlock / 1024.0 << "KB\n";
        std::cout << "每个SM的最大线程块数：" << devicePro.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "每个线程块的最大线程数：" << devicePro.maxThreadsPerBlock << std::endl;
        std::cout << "每个SM的最大线程数：" << devicePro.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "每个SM的最大线程束数：" << devicePro.warpSize << std::endl;
    }
}

namespace Color {
    inline Float3 random() {
        return color(Utils::randomFloat(), Utils::randomFloat(), Utils::randomFloat());
    }

    inline Float3 random(Float min, Float max) {
        auto randomColor = color(Utils::randomFloat(), Utils::randomFloat(), Utils::randomFloat());
        return (min + (max - min) * randomColor);
    }
}