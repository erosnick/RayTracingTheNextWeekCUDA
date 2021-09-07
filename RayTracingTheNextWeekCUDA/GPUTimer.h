#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CUDATypes.h"

#include <cstdio>
#include <string>
#include <iostream>

class GPUTimer {
public:
    GPUTimer(const std::string& message = "Start") {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        start(message);
    }

    void start(const std::string& message = "Start") {
        cudaEventRecord(startEvent, 0);
        printf("%s\n", message.c_str());
    }

    void stop(const std::string& message = "Stop") {
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&time, startEvent, stopEvent);
        printf("\n%s: %3.3f ms \n", message.c_str(), time);
    }

    float time = 0.0f;
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
};
