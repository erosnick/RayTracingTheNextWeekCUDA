#pragma once

#include "CUDATypes.h"
#include "HitResult.h"

class Hitable {
public:
    CUDA_DEVICE virtual bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const = 0;
};