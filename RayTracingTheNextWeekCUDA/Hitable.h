#pragma once

#include "CUDATypes.h"
#include "HitResult.h"

enum class PrimitiveType : uint8_t {
    Sphere,
    Plane
};

class Hitable {
public:
    CUDA_HOST_DEVICE virtual bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const = 0;
};