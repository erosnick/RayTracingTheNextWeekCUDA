#pragma once

#include "CUDATypes.h"
#include "HitResult.h"
#include "AABBox.h"

enum class PrimitiveType : uint8_t {
    Sphere,
    Plane,
    Triangle,
    TriangleMesh
};

class Hitable {
public:
    CUDA_DEVICE Hitable() {}
    CUDA_DEVICE virtual ~Hitable() {}
    CUDA_DEVICE virtual bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const = 0;
    CUDA_DEVICE virtual bool boundingBox(Float time0, Float time1, AABBox& outputAABB) const = 0;
};