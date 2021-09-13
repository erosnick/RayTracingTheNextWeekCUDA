#pragma once

#include "CUDATypes.h"
#include <cstdint>

class Ray {
public:
    CUDA_HOST_DEVICE inline Ray() {}
    CUDA_HOST_DEVICE inline Ray(const Float3& inOrigin, const Float3& inDirection, Float inTime = 0.0f)
        : origin(inOrigin), direction(inDirection), time(inTime) {
        inverseDirection = 1.0f / direction;
        signs[0] = (inverseDirection.x < 0.0f);
        signs[1] = (inverseDirection.y < 0.0f);
        signs[2] = (inverseDirection.z < 0.0f);
    }

    CUDA_HOST_DEVICE inline Float3 at(Float t) const {
        return origin + t * direction;
    }

    Float3 origin;
    Float3 direction;
    Float3 inverseDirection;
    int32_t signs[3];
    Float time;
};

