#pragma once

#include "CUDATypes.h"
#include <cstdint>
#include "LinearAlgebra.h"

class Ray {
public:
    CUDA_HOST_DEVICE inline Ray() {}
    CUDA_HOST_DEVICE inline Ray(const Vector3Df& inOrigin, const Vector3Df& inDirection, Float inTime = 0.0f)
        : origin(inOrigin), direction(inDirection), time(inTime) {
        inverseDirection = 1.0f / direction;
        signs[0] = (inverseDirection.x < 0.0f);
        signs[1] = (inverseDirection.y < 0.0f);
        signs[2] = (inverseDirection.z < 0.0f);
    }

    CUDA_HOST_DEVICE inline Vector3Df at(Float t) const {
        return origin + t * direction;
    }

    Vector3Df origin;           // ray origin
    Vector3Df direction;        // ray direction
    Vector3Df inverseDirection;
    int32_t signs[3];
    Float time;
};

