#pragma once

#include "CUDATypes.h"
#include "HitResult.h"

enum class PrimitiveType : uint8_t {
    Sphere,
    Plane,
    Triangle
};

class Hitable {
public:
    CUDA_DEVICE Hitable() {}
    CUDA_DEVICE virtual ~Hitable() {}
    CUDA_DEVICE virtual bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const = 0;
};